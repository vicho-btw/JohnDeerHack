"""
sim/grasp_env.py — PHASE 2: real physics grasp environment (MuJoCo + Franka Panda).

This is an ACTUAL rigid-body simulation (gravity, friction, contacts), not an
animation. A Franka Emika Panda (official mujoco_menagerie model) must reach a
free-floating part on a workbench, close its gripper, and lift it. A grasp
"succeeds" only if the part is physically raised above a height threshold by the
contact forces of the fingers — nothing is scripted to teleport.

The arm is driven by damped-least-squares (Jacobian) inverse kinematics toward a
sequence of Cartesian waypoints. The WAYPOINTS and gripper timing are functions
of a small parameter vector `theta` (the "policy"). `sim/train_grasp.py` learns
`theta` over thousands of these rollouts with the cross-entropy method, so the
behavior is genuinely optimized in simulation rather than hand-coded.

Tie to Phase 1: the part's location is the thing perception localizes, and the
task prompt is the observation_contract prompt ("pick up the part ..."). The
policy consumes that target the same way Pi-0 would consume an observation.

Interface:
    env = GraspEnv()
    env.reset(seed=...)                 # randomizes the part pose
    result = env.rollout(theta)         # dict: success, reward, lift, frames(optional)
"""
import os
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_HERE, "franka", "grasp_scene.xml")

# Home configuration: 7 arm joints + 2 finger joints (gripper open at 0.04).
HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04], dtype=np.float64)
# Matching position-servo targets (arm angles + gripper ctrl 255 == fully open).
HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255.0], dtype=np.float64)

# Offset from the `hand` body frame to the grasp center (between the fingertips),
# along the hand's local +z (the approach axis). Matches menagerie's gripper site.
GRIP_OFFSET = np.array([0.0, 0.0, 0.103], dtype=np.float64)

GRIP_OPEN = 255.0          # gripper actuator ctrl: fully open
PART_HALF_Z = 0.03         # half-height of the part box
LIFT_SUCCESS = 0.06        # part must rise this many meters to count as a grasp


# ---------------------------------------------------------------------------
# Policy parameterization (the vector the CEM learns)
# ---------------------------------------------------------------------------
# name           meaning                                          [low,    high]
PARAM_SPEC = [
    ("dx",             "grasp x offset vs part center (m)",       (-0.03, 0.03)),
    ("dy",             "grasp y offset vs part center (m)",       (-0.03, 0.03)),
    ("approach_z",     "hover height above part before descent",  (0.06,  0.16)),
    ("grasp_z",        "grip-center height at grasp (m)",         (0.02,  0.09)),
    ("yaw",            "gripper yaw to align jaw with part (rad)", (-0.8,  0.8)),
    ("lift_z",         "height to lift the part to (m)",          (0.20,  0.36)),
    ("close_ctrl",     "gripper close command (0=tight..255)",    (0.0,   140.0)),
    ("press",          "extra downward seat depth at grasp (m)",  (0.0,   0.05)),
]
PARAM_NAMES = [p[0] for p in PARAM_SPEC]
PARAM_LOW = np.array([p[2][0] for p in PARAM_SPEC], dtype=np.float64)
PARAM_HIGH = np.array([p[2][1] for p in PARAM_SPEC], dtype=np.float64)
N_PARAMS = len(PARAM_SPEC)
# A sensible (un-tuned) starting guess; CEM improves on this.
PARAM_INIT = np.array([0.0, 0.0, 0.10, 0.045, 0.0, 0.28, 40.0, 0.01], dtype=np.float64)


def clip_theta(theta):
    return np.clip(np.asarray(theta, dtype=np.float64), PARAM_LOW, PARAM_HIGH)


def describe_theta(theta):
    theta = clip_theta(theta)
    return {name: float(v) for name, v in zip(PARAM_NAMES, theta)}


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------
def _down_quat(yaw):
    """Quaternion for a gripper pointing straight down, rotated by `yaw` about z.

    Hand local +z is the approach axis; we want it along world -z. Build the
    target rotation matrix columns [x, y, z] and convert to a quaternion.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x_axis = np.array([c, s, 0.0])
    y_axis = np.array([s, -c, 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are the axes
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


class GraspEnv:
    def __init__(self, render_size=(480, 640), camera="demo"):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.hand_id = self.model.body("hand").id
        self.part_jnt = self.model.joint("part_free")
        self.part_qadr = int(self.part_jnt.qposadr[0])
        self.part_body = self.model.body("part").id
        # dof indices of the 7 arm joints (for restricting the IK Jacobian)
        self.arm_dofs = np.array(
            [int(self.model.joint(f"joint{i+1}").dofadr[0]) for i in range(7)])
        self.arm_qadr = np.array(
            [int(self.model.joint(f"joint{i+1}").qposadr[0]) for i in range(7)])
        self.arm_range = self.model.jnt_range[
            [self.model.joint(f"joint{i+1}").id for i in range(7)]]
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._ik_data = mujoco.MjData(self.model)  # scratch for kinematic IK
        self.render_size = render_size
        self.camera = camera
        self._renderer = None
        self.part_xy = np.array([0.5, 0.0])
        self.part_yaw = 0.0

    # --- part / state ------------------------------------------------------
    def reset(self, seed=None, part_xy=None, part_yaw=None):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = HOME_QPOS
        self.data.ctrl[:8] = HOME_CTRL
        if part_xy is None:
            part_xy = np.array([rng.uniform(0.46, 0.56), rng.uniform(-0.06, 0.06)])
        if part_yaw is None:
            part_yaw = rng.uniform(-0.5, 0.5)
        self.part_xy = np.asarray(part_xy, dtype=np.float64)
        self.part_yaw = float(part_yaw)
        q = self.data.qpos
        q[self.part_qadr + 0] = self.part_xy[0]
        q[self.part_qadr + 1] = self.part_xy[1]
        q[self.part_qadr + 2] = 0.005 + PART_HALF_Z  # resting on the bench pad
        cz, sz = np.cos(part_yaw / 2), np.sin(part_yaw / 2)
        q[self.part_qadr + 3:self.part_qadr + 7] = [cz, 0, 0, sz]  # yaw about z
        mujoco.mj_forward(self.model, self.data)
        return self._observe()

    def _observe(self):
        """Phase-1-style target: where perception says the part is (x, y, yaw)."""
        return {"part_xy": self.part_xy.copy(), "part_yaw": self.part_yaw,
                "prompt": "pick up the part and place it in the bin"}

    def grip_center(self):
        R = self.data.body(self.hand_id).xmat.reshape(3, 3)
        return self.data.body(self.hand_id).xpos + R @ GRIP_OFFSET

    def part_pos(self):
        return self.data.xpos[self.part_body].copy()

    # --- inverse kinematics ------------------------------------------------
    def _ik_solve(self, target_pos, target_quat, iters=150):
        """Kinematic damped-least-squares IK: return 7 arm joint angles that put
        the grip center at (target_pos, target_quat). Solved on a scratch MjData
        so the live simulation state is untouched (no integrator windup).

        Both the position and orientation errors are expressed in the WORLD frame
        to match mj_jac's jacp/jacr (also world frame). The orientation error is
        the axis-angle of R_target @ R_current^T.
        """
        ik = self._ik_data
        ik.qpos[:] = self.data.qpos  # warm start from the current pose
        R_des = np.zeros(9)
        mujoco.mju_quat2Mat(R_des, target_quat)
        R_des = R_des.reshape(3, 3)
        q = self.data.qpos[self.arm_qadr].copy()
        for _ in range(iters):
            ik.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.model, ik)
            mujoco.mj_comPos(self.model, ik)
            R = ik.body(self.hand_id).xmat.reshape(3, 3)
            grip = ik.body(self.hand_id).xpos + R @ GRIP_OFFSET
            pos_err = target_pos - grip
            # world-frame orientation error: axis-angle of R_des @ R_cur^T
            R_err = R_des @ R.T
            quat_err = np.zeros(4)
            mujoco.mju_mat2Quat(quat_err, R_err.flatten())
            angle = 2.0 * np.arccos(np.clip(quat_err[0], -1.0, 1.0))
            vnorm = np.linalg.norm(quat_err[1:])
            rot_err = (quat_err[1:] / vnorm) * angle if vnorm > 1e-9 else np.zeros(3)
            if np.linalg.norm(pos_err) < 8e-4 and np.linalg.norm(rot_err) < 8e-3:
                break
            err = np.concatenate([pos_err, 0.5 * rot_err])
            mujoco.mj_jac(self.model, ik, self._jacp, self._jacr, grip, self.hand_id)
            J = np.vstack([self._jacp[:, self.arm_dofs], self._jacr[:, self.arm_dofs]])
            lam = 0.1
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2),
                        self.arm_range[:, 0], self.arm_range[:, 1])
        return q

    def _run(self, target_pos, target_quat, grip_ctrl, n_steps, frames=None,
             frame_every=8, heights=None):
        """Move the arm to a Cartesian waypoint by solving IK once and ramping the
        position servos there over n_steps, while holding the gripper command."""
        q_goal = self._ik_solve(target_pos, target_quat)
        q_start = self.data.ctrl[:7].copy()
        for k in range(n_steps):
            alpha = (k + 1) / n_steps
            # smootherstep ease for gentle, contact-friendly motion
            a = alpha * alpha * alpha * (alpha * (alpha * 6 - 15) + 10)
            self.data.ctrl[:7] = (1 - a) * q_start + a * q_goal
            self.data.ctrl[7] = grip_ctrl
            mujoco.mj_step(self.model, self.data)
            if frames is not None and k % frame_every == 0:
                frames.append(self._render())
                if heights is not None:
                    heights.append(float(self.part_pos()[2]))
        return frames

    # --- rollout -----------------------------------------------------------
    def rollout(self, theta, record=False, frame_every=8):
        """Execute the staged grasp defined by `theta`. Returns metrics (and, if
        record=True, a list of rendered BGR frames for the dashboard)."""
        theta = clip_theta(theta)
        p = describe_theta(theta)
        frames = [] if record else None
        heights = [] if record else None

        target_xy = self.part_xy + np.array([p["dx"], p["dy"]])
        yaw = self.part_yaw + p["yaw"]
        quat = _down_quat(yaw)
        z0 = self.part_pos()[2]

        above = np.array([target_xy[0], target_xy[1], p["grasp_z"] + p["approach_z"]])
        at = np.array([target_xy[0], target_xy[1], p["grasp_z"]])
        seat = np.array([target_xy[0], target_xy[1], p["grasp_z"] - p["press"]])
        lifted = np.array([target_xy[0], target_xy[1], p["lift_z"]])
        bin_xy = np.array([0.45, -0.32])
        over_bin = np.array([bin_xy[0], bin_xy[1], p["lift_z"]])

        # Stage A: reach above the part, gripper open
        self._run(above, quat, GRIP_OPEN, 150, frames, frame_every, heights)
        # Stage B: descend onto the part (seat down by `press`)
        self._run(seat, quat, GRIP_OPEN, 140, frames, frame_every, heights)
        # Stage C: close the gripper (force-limited contact closes on the part)
        self._run(seat, quat, p["close_ctrl"], 90, frames, frame_every, heights)
        # Stage D: lift
        self._run(lifted, quat, p["close_ctrl"], 160, frames, frame_every, heights)

        lift = float(self.part_pos()[2] - z0)
        grip_miss = float(np.linalg.norm(self.grip_center()[:2] - self.part_pos()[:2]))
        success = lift > LIFT_SUCCESS

        if record:
            # Stage E (demo only): carry the part over the bin and release.
            self._run(over_bin, quat, p["close_ctrl"], 150, frames, frame_every, heights)
            self._run(over_bin, quat, GRIP_OPEN, 60, frames, frame_every, heights)

        # Shaped reward: dominated by how high the part was lifted, minus a small
        # penalty for missing the part horizontally. Smooth enough for CEM.
        reward = float(np.clip(lift, -0.02, 0.22) - 0.15 * grip_miss
                       + (0.05 if success else 0.0))

        return {"success": bool(success), "reward": reward, "lift": lift,
                "grip_miss": grip_miss, "frames": frames, "heights": heights,
                "z0": float(z0)}

    # --- rendering ---------------------------------------------------------
    def _render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
        self._renderer.update_scene(self.data, camera=self.camera)
        rgb = self._renderer.render()
        return rgb[:, :, ::-1].copy()  # BGR for cv2/dashboard

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def evaluate(theta, seeds, env=None):
    """Mean reward + success rate of `theta` over a set of random part placements."""
    own = env is None
    env = env or GraspEnv()
    rewards, succ, lifts = [], 0, []
    for s in seeds:
        env.reset(seed=int(s))
        r = env.rollout(theta)
        rewards.append(r["reward"])
        succ += int(r["success"])
        lifts.append(r["lift"])
    if own:
        env.close()
    return {"reward": float(np.mean(rewards)), "success_rate": succ / len(seeds),
            "mean_lift": float(np.mean(lifts))}


if __name__ == "__main__":
    # Smoke test: the un-tuned init policy should at least move and sometimes lift.
    env = GraspEnv()
    env.reset(seed=0)
    out = env.rollout(PARAM_INIT)
    print("init theta:", describe_theta(PARAM_INIT))
    print(f"rollout -> success={out['success']} lift={out['lift']:.3f} "
          f"reward={out['reward']:.3f} grip_miss={out['grip_miss']:.3f}")
    stats = evaluate(PARAM_INIT, seeds=range(6), env=env)
    print("init policy over 6 placements:", stats)
    env.close()
