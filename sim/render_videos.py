"""
sim/render_videos.py — render clean, labeled Phase 2 task videos.

For each task: pick a random layout, run the controller in its shipping mode,
capture frames with a HUD (task label, a Reach -> Grip -> Lift/Place stage
ribbon, live height, and a SUCCESS / constraint-assist badge), and write an MP4.

Outputs (data/):
    phase2_<task>.mp4   one per task
    phase2_grasp.mp4    == the lift task (named per the original success spec)
    phase2_all.mp4      all six tasks concatenated into one reel

Usage:
    python sim/render_videos.py                  # all tasks + reel
    python sim/render_videos.py --tasks stack    # just one
    python sim/render_videos.py --seed 5
"""
import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.manip_env import ManipEnv
from sim.tasks import TASKS, TASK_ORDER, run_task

FR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "franka")
TRACK = {"lift": "part", "bin_place": "part", "stack": "top",
         "bottle_cap": "cap", "peg_insert": "peg", "pyramid": "b2"}
W, H = 720, 540
RIBBON = ["Reach", "Grip", "Lift", "Move", "Place"]
STAGE_COLOR = {"Reach": (60, 170, 230), "Grip": (60, 200, 230), "Lift": (90, 200, 90),
               "Move": (200, 150, 60), "Place": (210, 160, 70), "Settle": (160, 160, 160),
               "Done": (90, 210, 110), "SUCCESS": (90, 210, 110)}


def _bar(img, y0, y1, a=0.5):
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (img.shape[1], y1), (16, 16, 20), -1)
    cv2.addWeighted(ov, a, img, 1 - a, 0, img)


def hud(frame, task, mode, stage, z_cm, seed, final=False, ok=False):
    f = frame.copy()
    _bar(f, 0, 66)
    cv2.putText(f, task.label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (245, 245, 248), 2, cv2.LINE_AA)
    cv2.putText(f, task.desc, (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (170, 200, 230), 1, cv2.LINE_AA)
    cv2.putText(f, mode, (W - 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 220, 150) if "REAL" in mode else (220, 200, 120), 1, cv2.LINE_AA)
    # stage ribbon (Reach -> Grip -> Lift -> Move -> Place), just under the top bar
    x = 14
    cur = stage if stage in RIBBON else ("Place" if stage in ("Settle", "Done", "SUCCESS") else stage)
    for i, s in enumerate(RIBBON):
        active = (s == cur)
        passed = RIBBON.index(cur) > i if cur in RIBBON else True
        col = (90, 200, 90) if (active or (final and ok)) else (
            (120, 130, 140) if passed else (70, 74, 82))
        cv2.putText(f, s, (x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        x += cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0] + 16
        if i < len(RIBBON) - 1:
            cv2.putText(f, ">", (x - 13, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (80, 84, 92), 1, cv2.LINE_AA)
    _bar(f, H - 40, H)
    cv2.putText(f, f"height: {z_cm:5.1f} cm    seed {seed}", (14, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
    chip = "SUCCESS" if (final and ok) else ("done" if final else stage)
    col = (90, 200, 90) if (final and ok) else STAGE_COLOR.get(stage, (150, 150, 150))
    (tw, _), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(f, (W - tw - 26, H - 34), (W - 12, H - 8), col, -1)
    cv2.putText(f, chip, (W - tw - 18, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (20, 20, 20), 2, cv2.LINE_AA)
    return f


def render_task(name, seed, hold_frames=40):
    task = TASKS[name]
    env = ManipEnv(os.path.join(FR, f"task_{task.scene}.xml"), render_size=(H, W))
    env.frame_every = 5
    mode = "VACUUM-ASSIST" if task.default_weld else "REAL GRASP"
    track = TRACK[name]
    frames = []

    def emit(e):
        z = float(e.body_pos(track)[2]) * 100
        frames.append(hud(e.render_frame(), task, mode, e.stage, z, seed))

    ok, z, text, _ = run_task(env, task, seed=seed, emit=emit)
    final = hud(env.render_frame(), task, mode, "SUCCESS" if ok else "done",
                z * 100, seed, final=True, ok=bool(ok))
    frames.extend([final] * hold_frames)
    env.close()
    return frames, bool(ok), z


def write_mp4(path, frames, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for fr in frames:
        vw.write(fr)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASK_ORDER)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--reel", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    reel = []
    summary = []
    for i, name in enumerate(args.tasks):
        seed = args.seed + i * 7
        frames, ok, z = render_task(name, seed)
        out = os.path.join(args.outdir, f"phase2_{name}.mp4")
        write_mp4(out, frames)
        dur = len(frames) / 30
        print(f"{name:11s} seed={seed} success={ok} z={z*100:.1f}cm "
              f"-> {out} ({dur:.1f}s, {len(frames)} frames)")
        summary.append((name, ok, z, out))
        if name == "lift":
            write_mp4(os.path.join(args.outdir, "phase2_grasp.mp4"), frames)
        reel.extend(frames)

    if args.reel and len(args.tasks) > 1:
        reel_path = os.path.join(args.outdir, "phase2_all.mp4")
        write_mp4(reel_path, reel)
        print(f"reel -> {reel_path} ({len(reel)/30:.1f}s)")
    return summary


if __name__ == "__main__":
    main()
