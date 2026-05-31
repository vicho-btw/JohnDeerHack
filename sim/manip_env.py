"""
sim/manip_env.py — generalized Franka manipulation env (Phase 2, multi-task).

One env, many tasks. Loads a task scene (Franka Panda + task objects), exposes
Cartesian manipulation primitives (move_to / open_grip / close_grip / pick /
place) built on damped-least-squares Jacobian IK, and an optional weld-assist
("vacuum/magnetic" grip) used as the documented fallback when frictional grasp is
unreliable for a given object.

The control primitives take an `emit` callback that is invoked on each rendered
frame, with the env carrying a live `stage` label and `metric`. This single code
path feeds BOTH the offline video renderer and the live web dashboard.

Scenes are built by sim/scenes.py and written to sim/franka/task_*.xml so the
`<include file="panda.xml"/>` and mesh assets resolve. Movable objects have free
joints and are randomized per seed at reset().
"""
import os
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
FRANKA_DIR = os.path.join(_HERE, "franka")

HOME_ARM = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853], dtype=np.float64)
GRIP_OFFSET = np.array([0.0, 0.0, 0.103], dtype=np.float64)  # hand frame -> grasp center
GRIP_OPEN = 255.0
GRIP_CLOSE = 18.0     # learned-ish close command (force-limited)


def down_quat(yaw):
    """Quaternion for a gripper pointing straight down, rotated `yaw` about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.stack([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


class ManipEnv:
    def __init__(self, xml_path, camera="demo", render_size=(480, 640)):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.hand_id = self.model.body("hand").id
        self.arm_qadr = np.array(
            [int(self.model.joint(f"joint{i+1}").qposadr[0]) for i in range(7)])
        self.arm_dofs = np.array(
            [int(self.model.joint(f"joint{i+1}").dofadr[0]) for i in range(7)])
        self.arm_range = self.model.jnt_range[
            [self.model.joint(f"joint{i+1}").id for i in range(7)]]
        self.finger_qadr = [int(self.model.joint("finger_joint1").qposadr[0]),
                            int(self.model.joint("finger_joint2").qposadr[0])]
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._ik = mujoco.MjData(self.model)
        # only our weld equalities are runtime-toggled (NOT panda's finger coupling)
        self.weld_ids = [i for i in range(self.model.neq)
                         if (self.model.equality(i).name or "").startswith("weld_")]
        self.render_size = render_size
        self.camera = camera
        self._renderer = None
        self.stage = "idle"
        self.metric = 0.0
        self.frame_every = 7

    # --- body / object helpers --------------------------------------------
    def body_pos(self, name):
        return self.data.body(name).xpos.copy()

    def body_quat(self, name):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.body(name).xmat)
        return q

    def set_free_body(self, joint_name, pos, yaw=0.0):
        adr = int(self.model.joint(joint_name).qposadr[0])
        self.data.qpos[adr:adr + 3] = pos
        cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
        self.data.qpos[adr + 3:adr + 7] = [cz, 0, 0, sz]

    def grip_center(self):
        R = self.data.body(self.hand_id).xmat.reshape(3, 3)
        return self.data.body(self.hand_id).xpos + R @ GRIP_OFFSET

    def finger_gap(self):
        return float(self.data.qpos[self.finger_qadr[0]] + self.data.qpos[self.finger_qadr[1]])

    # --- reset -------------------------------------------------------------
    def home(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = HOME_ARM
        self.data.qpos[self.finger_qadr[0]] = 0.04
        self.data.qpos[self.finger_qadr[1]] = 0.04
        for i in range(7):
            self.data.ctrl[i] = HOME_ARM[i]
        self.data.ctrl[7] = GRIP_OPEN
        # deactivate only our weld assists (keep panda's finger-coupling equality)
        for i in self.weld_ids:
            self.data.eq_active[i] = 0
        mujoco.mj_forward(self.model, self.data)

    # --- inverse kinematics (world-frame DLS, kinematic on scratch data) ---
    def ik_solve(self, target_pos, target_quat, iters=150):
        ik = self._ik
        ik.qpos[:] = self.data.qpos
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
            R_err = R_des @ R.T
            qd = np.zeros(4)
            mujoco.mju_mat2Quat(qd, R_err.flatten())
            ang = 2.0 * np.arccos(np.clip(qd[0], -1.0, 1.0))
            vn = np.linalg.norm(qd[1:])
            rot_err = (qd[1:] / vn) * ang if vn > 1e-9 else np.zeros(3)
            if np.linalg.norm(pos_err) < 8e-4 and np.linalg.norm(rot_err) < 8e-3:
                break
            err = np.concatenate([pos_err, 0.5 * rot_err])
            mujoco.mj_jac(self.model, ik, self._jacp, self._jacr, grip, self.hand_id)
            J = np.vstack([self._jacp[:, self.arm_dofs], self._jacr[:, self.arm_dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2),
                        self.arm_range[:, 0], self.arm_range[:, 1])
        return q

    # --- weld assist (fallback grip) --------------------------------------
    def weld_attach(self, weld_name):
        """Freeze the welded object at its CURRENT relative pose to the hand."""
        eid = self.model.equality(weld_name).id
        b1 = self.model.eq_obj1id[eid]
        b2 = self.model.eq_obj2id[eid]
        p1, p2 = self.data.xpos[b1], self.data.xpos[b2]
        q1 = np.zeros(4); q2 = np.zeros(4)
        mujoco.mju_mat2Quat(q1, self.data.xmat[b1])
        mujoco.mju_mat2Quat(q2, self.data.xmat[b2])
        q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, q1)
        relpos = np.zeros(3)
        mujoco.mju_rotVecQuat(relpos, p2 - p1, q1inv)  # p2 in body1 frame
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, q1inv, q2)
        self.model.eq_data[eid, 0:3] = 0.0          # anchor (body2 frame)
        self.model.eq_data[eid, 3:6] = relpos
        self.model.eq_data[eid, 6:10] = relquat
        self.model.eq_data[eid, 10] = 1.0           # torquescale
        self.data.eq_active[eid] = 1

    def weld_release(self, weld_name):
        eid = self.model.equality(weld_name).id
        self.data.eq_active[eid] = 0

    # --- primitives --------------------------------------------------------
    def _emit(self, emit):
        if emit is not None:
            emit(self)

    def move_to(self, pos, quat, n_steps, grip_ctrl, emit=None, stage=None,
                settle_only=False):
        if stage:
            self.stage = stage
        q_goal = self.ik_solve(pos, quat) if not settle_only else self.data.ctrl[:7].copy()
        q_start = self.data.ctrl[:7].copy()
        for k in range(n_steps):
            a = (k + 1) / n_steps
            a = a * a * a * (a * (a * 6 - 15) + 10)   # smootherstep
            self.data.ctrl[:7] = (1 - a) * q_start + a * q_goal
            self.data.ctrl[7] = grip_ctrl
            mujoco.mj_step(self.model, self.data)
            if k % self.frame_every == 0:
                self._emit(emit)

    def settle(self, n_steps, grip_ctrl, emit=None, stage=None):
        if stage:
            self.stage = stage
        for k in range(n_steps):
            self.data.ctrl[7] = grip_ctrl
            mujoco.mj_step(self.model, self.data)
            if k % self.frame_every == 0:
                self._emit(emit)

    # --- rendering ---------------------------------------------------------
    def render_frame(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
        self._renderer.update_scene(self.data, camera=self.camera)
        rgb = self._renderer.render()
        return rgb[:, :, ::-1].copy()  # BGR

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
