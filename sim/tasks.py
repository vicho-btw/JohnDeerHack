"""
sim/tasks.py — the 6 Phase 2 manipulation tasks + their controllers.

Each task: a randomized reset, a staged Cartesian controller built from
manip_env primitives (pick / place / move), and a physics-based success check
(z-height / placement tolerance). Controllers take an `emit` callback so the same
code drives both the offline video renderer and the live dashboard.

`weld` flag: try real frictional grasp; if a task proves unreliable it is run with
weld-assist (MuJoCo equality weld == "vacuum/magnetic" grip). Which mode each task
ships in is decided by sim/verify_tasks.py and recorded in STATUS.md.
"""
import numpy as np

from sim.manip_env import ManipEnv, down_quat, GRIP_OPEN, GRIP_CLOSE
from sim.scenes import (BLOCK, CAP_R, CAP_H, BOTTLE_H, PEG_HZ, HOLE, BENCH_Z)

GRASP_ABOVE = 0.012     # grip center sits this far above an object's mid-height
APPROACH = 0.11         # hover height above grasp before descending


def yaw_of(env, name):
    q = env.body_quat(name)
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                            1 - 2 * (q[2] ** 2 + q[3] ** 2)))


# ---- generic primitives ----------------------------------------------------
def pick(env, obj, emit, weld=False, grip_yaw=None, n=(150, 130, 90, 110),
         grasp_above=GRASP_ABOVE):
    """Reach above, descend, grip, lift. With weld-assist the weld is attached
    BEFORE the fingers close, so closing can't knock thin/small objects out of
    position. Returns (quat, offset) where offset = grip_center - object_center
    after lifting; place() uses it to land the object exactly on target."""
    pos = env.body_pos(obj)
    yaw = yaw_of(env, obj) if grip_yaw is None else grip_yaw
    quat = down_quat(yaw)
    gz = pos[2] + grasp_above
    above = np.array([pos[0], pos[1], gz + APPROACH])
    at = np.array([pos[0], pos[1], gz])
    # weld-assist == vacuum grip: keep the fingers OPEN so they never touch (and
    # so can't drag) the object; the weld does the holding. Real grasp closes.
    hold = GRIP_OPEN if weld else GRIP_CLOSE
    env.move_to(above, quat, n[0], GRIP_OPEN, emit, "Reach")
    env.move_to(at, quat, n[1], GRIP_OPEN, emit, "Grip")
    env.settle(30, GRIP_OPEN, emit, "Grip")     # converge to grasp depth (kill servo lag)
    if weld:
        env.weld_attach(f"weld_{obj}")
    env.settle(n[2], hold, emit, "Grip")        # real grasp closes; vacuum stays open
    lift = np.array([pos[0], pos[1], gz + 0.20])
    env.move_to(lift, quat, n[3], hold, emit, "Lift")
    env.settle(20, hold, emit, "Lift")
    offset = env.grip_center() - env.body_pos(obj)   # how the object hangs in-grip
    return quat, offset, hold


def place(env, obj, dest, emit, quat, offset, hold, weld=False, drop_gap=0.006,
          n=(130, 110, 70), open_after=True):
    """Carry the object so its center lands at `dest`, then release. `offset`
    (from pick) maps object pose -> grip-center pose, making placement exact.
    `hold` is the gripper command kept during the carry (open for vacuum/weld)."""
    dest = np.asarray(dest, dtype=float)
    # Approach directly over the target (only the z offset matters up here), then
    # settle so a swinging/lagging object hangs straight down, and RE-MEASURE the
    # grasp offset. This cancels any sideways lag from the horizontal carry.
    above = np.array([dest[0], dest[1], dest[2] + offset[2] + APPROACH + 0.04])
    env.move_to(above, quat, n[0], hold, emit, "Move")
    env.settle(45, hold, emit, "Move")
    offset = env.grip_center() - env.body_pos(obj)
    at = dest + offset + np.array([0, 0, drop_gap])
    env.move_to(at, quat, n[1], hold, emit, "Place")
    env.settle(55, hold, emit, "Place")         # converge onto target (kill overshoot)
    if weld:
        env.weld_release(f"weld_{obj}")
    if open_after:
        env.settle(n[2], GRIP_OPEN, emit, "Place")     # open fully (clear the object)
        # retreat straight up from where the gripper actually is, so the path
        # never brushes the just-placed object sideways
        up = env.grip_center() + np.array([0, 0, 0.18])
        env.move_to(up, quat, 110, GRIP_OPEN, emit, "Done")


def _rng_xy(rng, x=(0.46, 0.56), y=(-0.06, 0.06)):
    return np.array([rng.uniform(*x), rng.uniform(*y)])


# ---- task definitions ------------------------------------------------------
class Task:
    def __init__(self, name, label, desc, criterion, scene, default_weld=False):
        self.name, self.label, self.desc = name, label, desc
        self.criterion, self.scene = criterion, scene
        self.default_weld = default_weld

    def reset(self, env, rng):  # -> ctx
        raise NotImplementedError

    def control(self, env, ctx, emit, weld):
        raise NotImplementedError

    def success(self, env, ctx):  # -> (bool, z_metric, text)
        raise NotImplementedError


class Lift(Task):
    def reset(self, env, rng):
        env.home()
        xy = _rng_xy(rng)
        env.set_free_body("part_free", [xy[0], xy[1], BLOCK + BENCH_Z],
                          rng.uniform(-0.5, 0.5))
        import mujoco; mujoco.mj_forward(env.model, env.data)
        return {"z0": env.body_pos("part")[2]}

    def control(self, env, ctx, emit, weld):
        _q, _o, hold = pick(env, "part", emit, weld)
        env.settle(40, hold, emit, "Lift")

    def success(self, env, ctx):
        z = env.body_pos("part")[2]
        return z - ctx["z0"] > 0.06, z, f"part lifted to z={z * 100:.1f} cm"


class BinPlace(Task):
    def reset(self, env, rng):
        env.home()
        xy = _rng_xy(rng, x=(0.46, 0.56), y=(0.0, 0.1))
        env.set_free_body("part_free", [xy[0], xy[1], BLOCK + BENCH_Z],
                          rng.uniform(-0.5, 0.5))
        import mujoco; mujoco.mj_forward(env.model, env.data)
        return {"bin": np.array([0.5, -0.2])}

    def control(self, env, ctx, emit, weld):
        quat, off, hold = pick(env, "part", emit, weld)
        bx, by = ctx["bin"]
        place(env, "part", [bx, by, BENCH_Z + BLOCK], emit, quat, off, hold, weld, drop_gap=0.04)

    def success(self, env, ctx):
        p = env.body_pos("part")
        inb = abs(p[0] - ctx["bin"][0]) < 0.06 and abs(p[1] - ctx["bin"][1]) < 0.06
        return inb and p[2] < 0.09, p[2], f"part in bin (x={p[0]:.2f}, y={p[1]:.2f})"


class Stack(Task):
    def reset(self, env, rng):
        env.home()
        base = _rng_xy(rng, x=(0.47, 0.55), y=(-0.12, -0.04))
        top = _rng_xy(rng, x=(0.47, 0.55), y=(0.04, 0.12))
        env.set_free_body("base_free", [base[0], base[1], BLOCK + BENCH_Z], 0.0)
        env.set_free_body("top_free", [top[0], top[1], BLOCK + BENCH_Z],
                          rng.uniform(-0.4, 0.4))
        import mujoco; mujoco.mj_forward(env.model, env.data)
        return {}

    def control(self, env, ctx, emit, weld):
        base = env.body_pos("base")
        quat, off, hold = pick(env, "top", emit, weld)
        tgt = [base[0], base[1], base[2] + 2 * BLOCK]  # one block-height up
        place(env, "top", tgt, emit, quat, off, hold, weld, drop_gap=0.008)

    def success(self, env, ctx):
        base, top = env.body_pos("base"), env.body_pos("top")
        aligned = abs(top[0] - base[0]) < 0.02 and abs(top[1] - base[1]) < 0.02
        stacked = top[2] - base[2] > 1.5 * BLOCK
        return aligned and stacked, top[2], f"top z={top[2]*100:.1f} cm over base"


class BottleCap(Task):
    def reset(self, env, rng):
        env.home()
        cap = _rng_xy(rng, x=(0.47, 0.55), y=(0.05, 0.12))
        env.set_free_body("cap_free", [cap[0], cap[1], CAP_H + BENCH_Z], 0.0)
        import mujoco; mujoco.mj_forward(env.model, env.data)
        return {"bottle": np.array([0.5, -0.06])}

    def control(self, env, ctx, emit, weld):
        quat, off, hold = pick(env, "cap", emit, weld, grip_yaw=0.0)
        bx, by = ctx["bottle"]
        # Seat the cap right at rest height; the re-measure + converge-hold in
        # place() land it centred so it settles instead of bouncing off.
        place(env, "cap", [bx, by, BOTTLE_H + CAP_H], emit, quat, off, hold,
              weld, drop_gap=0.0, n=(150, 160, 55))

    def success(self, env, ctx):
        c = env.body_pos("cap")
        bx, by = ctx["bottle"]
        on = abs(c[0] - bx) < 0.03 and abs(c[1] - by) < 0.03 and c[2] > BOTTLE_H - 0.02
        return on, c[2], f"cap z={c[2]*100:.1f} cm on bottle"


class PegInsert(Task):
    def reset(self, env, rng):
        env.home()
        peg = _rng_xy(rng, x=(0.47, 0.55), y=(0.05, 0.12))
        env.set_free_body("peg_free", [peg[0], peg[1], PEG_HZ + BENCH_Z], 0.0)
        import mujoco; mujoco.mj_forward(env.model, env.data)
        return {"hole": np.array([0.5, -0.08])}

    def control(self, env, ctx, emit, weld):
        quat, off, hold = pick(env, "peg", emit, weld, grip_yaw=0.0, n=(150, 130, 90, 130))
        hx, hy = ctx["hole"]
        # lower the peg so its lower half enters the socket
        place(env, "peg", [hx, hy, PEG_HZ + BENCH_Z], emit, quat, off, hold,
              weld, drop_gap=0.002, n=(140, 160, 80))

    def success(self, env, ctx):
        p = env.body_pos("peg")
        hx, hy = ctx["hole"]
        centered = abs(p[0] - hx) < 0.02 and abs(p[1] - hy) < 0.02
        seated = p[2] < PEG_HZ + BENCH_Z + 0.02   # peg dropped into the hole
        return centered and seated, p[2], f"peg in socket (z={p[2]*100:.1f} cm)"


class Pyramid(Task):
    def reset(self, env, rng):
        env.home()
        cx = rng.uniform(0.49, 0.53)
        for b, x in zip(["b0", "b1", "b2"], [0.45, 0.50, 0.55]):
            xy = [x + rng.uniform(-0.01, 0.01), 0.11 + rng.uniform(-0.01, 0.01)]
            env.set_free_body(f"{b}_free", [xy[0], xy[1], BLOCK + BENCH_Z], 0.0)
        import mujoco; mujoco.mj_forward(env.model, env.data)
        d = 1.02 * BLOCK
        return {"cx": cx, "base_y": -0.07, "d": d}

    def control(self, env, ctx, emit, weld):
        cx, by, d = ctx["cx"], ctx["base_y"], ctx["d"]
        z1 = BLOCK + BENCH_Z
        z2 = z1 + 2 * BLOCK
        # two base blocks side by side, then the apex on top between them
        q, o, h = pick(env, "b0", emit, weld); place(env, "b0", [cx - d, by, z1], emit, q, o, h, weld)
        q, o, h = pick(env, "b1", emit, weld); place(env, "b1", [cx + d, by, z1], emit, q, o, h, weld)
        q, o, h = pick(env, "b2", emit, weld); place(env, "b2", [cx, by, z2], emit, q, o, h, weld,
                                                     drop_gap=0.006)

    def success(self, env, ctx):
        b2 = env.body_pos("b2")
        up = b2[2] > BLOCK + BENCH_Z + 1.5 * BLOCK
        centered = abs(b2[0] - ctx["cx"]) < 0.03 and abs(b2[1] - ctx["base_y"]) < 0.04
        return up and centered, b2[2], f"apex z={b2[2]*100:.1f} cm"


TASKS = {
    t.name: t for t in [
        Lift("lift", "Pick & Lift", "Grasp the block and lift it clear of the bench.",
             "block raised > 6 cm", "lift"),
        BinPlace("bin_place", "Pick & Place in Bin", "Grasp the block and drop it into the bin.",
                 "block resting inside the bin", "bin_place"),
        Stack("stack", "Stack Two Blocks", "Pick the orange block and stack it on the blue one.",
              "orange block stacked & aligned on blue", "stack"),
        BottleCap("bottle_cap", "Cap the Bottle", "Pick the cap and seat it on the bottle.",
                  "cap seated on the bottle top", "bottle_cap", default_weld=True),
        PegInsert("peg_insert", "Peg-in-Hole", "Pick the peg and insert it into the socket.",
                  "peg inserted into the socket", "peg_insert", default_weld=True),
        Pyramid("pyramid", "Build a Pyramid", "Stack three blocks into a 2-1 pyramid.",
                "apex block placed on the two-block base", "pyramid"),
    ]
}
TASK_ORDER = ["lift", "bin_place", "stack", "bottle_cap", "peg_insert", "pyramid"]


def run_task(env, task, seed, emit=None, weld=None):
    """Reset + control + settle + success. Returns (ok, z, text, ctx).
    weld=None uses the task's shipping default (task.default_weld)."""
    if weld is None:
        weld = task.default_weld
    rng = np.random.default_rng(seed)
    ctx = task.reset(env, rng)
    task.control(env, ctx, emit, weld)
    env.stage = "Settle"
    env.settle(60, GRIP_OPEN, emit, "Settle")
    ok, z, text = task.success(env, ctx)
    env.stage = "SUCCESS" if ok else "done"
    return ok, z, text, ctx
