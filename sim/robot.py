"""
sim/robot.py — PISTA de integracion (tu Mac, CPU, sin GPU).

Carga un brazo Franka Panda en PyBullet y le hace ejecutar las trayectorias que
Pi-0 genero. Pi-0 produce 14-D (2 brazos x 7 joints); el Franka tiene 7 joints,
asi que por defecto usamos los primeros 7 valores (un brazo). Esto basta para el demo.

Produce el video del PANEL CENTRAL del dashboard.

Uso (con GUI para verlo en vivo):
    python sim/robot.py --traj data/trajectories.pkl --gui

Uso (headless, graba frames para el dashboard):
    python sim/robot.py --traj data/trajectories.pkl --out data/sim_frames.pkl
"""
import argparse
import pickle
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observation_contract import HORIZON, STATE_DIM


def normalize_traj(traj):
    """Acepta (50,14) o (14,50) y devuelve (T, 14)."""
    traj = np.asarray(traj, dtype=np.float32)
    if traj.shape == (STATE_DIM, HORIZON):
        traj = traj.T
    return traj  # (T, 14)


def setup_sim(gui=False):
    import pybullet as p
    import pybullet_data
    mode = p.GUI if gui else p.DIRECT
    cid = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    # joints controlables del Panda (7 de brazo)
    arm_joints = [j for j in range(p.getNumJoints(robot))
                  if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE][:7]
    return p, robot, arm_joints, cid


def capture_frame(p, w=480, h=480):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.4], distance=1.6,
        yaw=50, pitch=-30, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=5)
    _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj)
    return np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)


def execute(traj, gui=False, steps_per_action=8, capture=True):
    """
    Ejecuta una trayectoria de joints en el Franka.
    Solo usamos los primeros 7 valores de cada paso (un brazo).
    Devuelve lista de frames RGB si capture=True.
    """
    p, robot, arm_joints, cid = setup_sim(gui)
    traj = normalize_traj(traj)
    frames = []
    for step in traj:
        targets = step[:7]
        for ji, jval in zip(arm_joints, targets):
            p.setJointMotorControl2(robot, ji, p.POSITION_CONTROL,
                                    targetPosition=float(jval), force=200)
        for _ in range(steps_per_action):
            p.stepSimulation()
            if gui:
                time.sleep(1 / 240.0)
        if capture and not gui:
            frames.append(capture_frame(p))
    p.disconnect(cid)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="pickle de call_policy")
    ap.add_argument("--gui", action="store_true", help="ventana en vivo (para verlo tu)")
    ap.add_argument("--out", default="data/sim_frames.pkl")
    ap.add_argument("--index", type=int, default=0, help="cual trayectoria ejecutar")
    args = ap.parse_args()

    with open(args.traj, "rb") as f:
        trajectories = pickle.load(f)["trajectories"]
    traj = trajectories[args.index]
    print(f"ejecutando trayectoria {args.index}, shape {np.asarray(traj).shape}")

    frames = execute(traj, gui=args.gui)
    if not args.gui:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump({"frames": frames}, f)
        print(f"guardados {len(frames)} frames del sim en {args.out}")
    else:
        print("ejecucion con GUI terminada")


if __name__ == "__main__":
    main()
