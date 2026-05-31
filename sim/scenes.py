"""
sim/scenes.py — MJCF scene builders for the Phase 2 tasks.

Each task scene = Franka Panda (included) + a shared workcell (floor, bench,
lights, cameras) + task-specific objects. Movable objects get a free joint AND a
(initially inactive) weld equality to the hand, so a task can fall back to
weld-assisted "vacuum" grip if frictional grasp is unreliable. Fixtures (bins,
bottle, socket) are static geoms.

Object START poses are nominal here; tasks.py randomizes them per seed at reset.
Files are written into sim/franka/ so `<include file="panda.xml"/>` + mesh assets
resolve.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
FRANKA_DIR = os.path.join(_HERE, "franka")

# ---- shared geometry constants (metres) ----
BLOCK = 0.022          # block half-size (cube) -> 44 mm wide, fits the 80 mm gripper
CAP_R, CAP_H = 0.018, 0.022      # bottle cap: short graspable cylinder (half-height)
BOTTLE_R, BOTTLE_H = 0.03, 0.075  # static bottle (short jar for gripper clearance)
PEG_HX, PEG_HZ = 0.011, 0.05     # peg half-width / half-length
HOLE = 0.018           # socket inner half-width (peg 11mm half -> 7mm clearance)
BENCH_Z = 0.005        # bench pad top

_HEADER = """<mujoco model="franka {name}">
  <include file="panda.xml"/>
  <statistic center="0.42 0 0.2" extent="0.9"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-22" offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.35 0.5" rgb2="0.02 0.02 0.05" width="512" height="3072"/>
    <texture type="2d" name="tk_grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.12 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="tk_grid" texture="tk_grid" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="tk_bench" rgba="0.45 0.42 0.38 1"/>
    <material name="tk_orange" rgba="0.95 0.55 0.1 1"/>
    <material name="tk_blue"   rgba="0.2 0.45 0.85 1"/>
    <material name="tk_green"  rgba="0.2 0.7 0.35 1"/>
    <material name="tk_red"    rgba="0.85 0.2 0.2 1"/>
    <material name="tk_cap"    rgba="0.9 0.75 0.15 1"/>
    <material name="tk_bottle" rgba="0.3 0.55 0.75 0.55"/>
    <material name="tk_steel"  rgba="0.6 0.62 0.66 1"/>
    <material name="tk_bin"    rgba="0.28 0.32 0.4 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 1.6" dir="0 0 -1" directional="true"/>
    <light pos="0.1 0.5 1.1" dir="0.2 -0.4 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="tk_grid"/>
    <geom name="tk_bench" type="box" pos="0.5 0 0.0025" size="0.22 0.22 0.0025" material="tk_bench"/>
"""

_CAMERAS = """    <camera name="demo" pos="1.25 -0.95 0.8" xyaxes="0.62 0.78 0 -0.3 0.24 0.92"/>
    <camera name="side" pos="0.5 -1.1 0.45" xyaxes="1 0 0 0 0.35 0.94"/>
    <camera name="top"  pos="0.5 0 1.3" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
"""


def _movable(name, geom, material, pos, mass=0.05, friction="1.4 0.06 0.002"):
    return f"""    <body name="{name}" pos="{pos}">
      <freejoint name="{name}_free"/>
      <geom name="{name}" {geom} material="{material}" mass="{mass}"
        friction="{friction}" condim="4" solref="0.008 1" solimp="0.95 0.99 0.001"/>
    </body>
"""


def _welds(names):
    w = "  <equality>\n"
    for n in names:
        w += (f'    <weld name="weld_{n}" body1="hand" body2="{n}" active="false"'
              f' solref="0.008 1" solimp="0.97 0.99 0.001"/>\n')
    w += "  </equality>\n"
    return w


def _box(hx, hy, hz):
    return f'type="box" size="{hx} {hy} {hz}"'


def _cyl(r, h):
    return f'type="cylinder" size="{r} {h}"'


# ---- per-task object blocks ----
def _lift_objects():
    return _movable("part", _box(BLOCK, BLOCK, BLOCK), "tk_orange", "0.5 0 0.03"), ["part"]


def _bin_objects():
    objs = _movable("part", _box(BLOCK, BLOCK, BLOCK), "tk_orange", "0.5 0.05 0.03")
    # bin: 4 walls forming a 0.13 x 0.13 open box at x~0.5, y~-0.18
    bx, by, wall, inner, wz = 0.5, -0.2, 0.006, 0.06, 0.035
    objs += f'    <geom name="bin_f" type="box" pos="{bx} {by-inner} {wz}" size="{inner+wall} {wall} {wz}" material="tk_bin"/>\n'
    objs += f'    <geom name="bin_b" type="box" pos="{bx} {by+inner} {wz}" size="{inner+wall} {wall} {wz}" material="tk_bin"/>\n'
    objs += f'    <geom name="bin_l" type="box" pos="{bx-inner} {by} {wz}" size="{wall} {inner} {wz}" material="tk_bin"/>\n'
    objs += f'    <geom name="bin_r" type="box" pos="{bx+inner} {by} {wz}" size="{wall} {inner} {wz}" material="tk_bin"/>\n'
    return objs, ["part"]


def _stack_objects():
    objs = _movable("base", _box(BLOCK, BLOCK, BLOCK), "tk_blue", "0.5 -0.08 0.03", mass=0.12)
    objs += _movable("top", _box(BLOCK, BLOCK, BLOCK), "tk_orange", "0.5 0.08 0.03")
    return objs, ["base", "top"]


def _bottle_objects():
    # static bottle (fixture) + movable cap
    objs = f'    <geom name="tk_bottle" {_cyl(BOTTLE_R, BOTTLE_H/2)} pos="0.5 -0.06 {BOTTLE_H/2}" material="tk_bottle"/>\n'
    objs += _movable("cap", _cyl(CAP_R, CAP_H), "tk_cap", "0.5 0.08 0.027", mass=0.03)
    return objs, ["cap"]


def _peg_objects():
    # static socket: 4 walls around a square hole at x~0.5,y~-0.08, on the bench
    sx, sy, wall, sz = 0.5, -0.08, 0.012, 0.03
    o = HOLE + wall
    objs = f'    <geom name="sock_f" type="box" pos="{sx} {sy-o} {sz}" size="{o+wall} {wall} {sz}" material="tk_steel"/>\n'
    objs += f'    <geom name="sock_b" type="box" pos="{sx} {sy+o} {sz}" size="{o+wall} {wall} {sz}" material="tk_steel"/>\n'
    objs += f'    <geom name="sock_l" type="box" pos="{sx-o} {sy} {sz}" size="{wall} {o} {sz}" material="tk_steel"/>\n'
    objs += f'    <geom name="sock_r" type="box" pos="{sx+o} {sy} {sz}" size="{wall} {o} {sz}" material="tk_steel"/>\n'
    objs += _movable("peg", _box(PEG_HX, PEG_HX, PEG_HZ), "tk_red", "0.5 0.08 0.05", mass=0.04)
    return objs, ["peg"]


def _pyramid_objects():
    objs = _movable("b0", _box(BLOCK, BLOCK, BLOCK), "tk_blue", "0.46 0.10 0.03")
    objs += _movable("b1", _box(BLOCK, BLOCK, BLOCK), "tk_green", "0.50 0.12 0.03")
    objs += _movable("b2", _box(BLOCK, BLOCK, BLOCK), "tk_orange", "0.54 0.10 0.03")
    return objs, ["b0", "b1", "b2"]


BUILDERS = {
    "lift": _lift_objects,
    "bin_place": _bin_objects,
    "stack": _stack_objects,
    "bottle_cap": _bottle_objects,
    "peg_insert": _peg_objects,
    "pyramid": _pyramid_objects,
}


def build_scene(name):
    objs, movable = BUILDERS[name]()
    xml = _HEADER.format(name=name) + objs + _CAMERAS + _welds(movable) + "</mujoco>\n"
    return xml


def write_all():
    paths = {}
    for name in BUILDERS:
        p = os.path.join(FRANKA_DIR, f"task_{name}.xml")
        with open(p, "w") as f:
            f.write(build_scene(name))
        paths[name] = p
    return paths


if __name__ == "__main__":
    import mujoco
    paths = write_all()
    for name, p in paths.items():
        m = mujoco.MjModel.from_xml_path(p)
        print(f"{name:11s} OK  nq={m.nq:2d} nu={m.nu} neq={m.neq}  -> {os.path.basename(p)}")
