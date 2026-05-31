"""
sim/run_grasp.py — PHASE 2 deliverable: render the LEARNED grasp policy executing
in the real MuJoCo simulation.

Loads the policy learned by sim/train_grasp.py (data/grasp_policy.npz) and runs it
on fresh, randomized part placements it never saw during training, rendering the
Franka picking the part and placing it in the bin. A HUD reports the live part
height and grasp status, and the headline video pairs the training learning curve
with the live execution — the whole Phase 2 story in one frame.

Tie to Phase 1: the task prompt comes straight from the observation_contract
(the same instruction Pi-0 receives). The part pose is what perception localizes.

Outputs:
    data/sim_frames.pkl    raw BGR frames ({"frames": [...]}) — dashboard-compatible
    data/grasp_sim.mp4     the grasp with HUD
    data/PHASE2_DEMO.mp4   2-panel: learning curve | live learned grasp  (headline)

Usage:
    python sim/run_grasp.py                      # 3 attempts, default policy
    python sim/run_grasp.py --attempts 4 --seed 3
"""
import argparse
import os
import pickle
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.grasp_env import GraspEnv, LIFT_SUCCESS, PARAM_INIT, describe_theta
from observation_contract import DEFAULT_PROMPT

W, H = 640, 480


def _bar(img, y0, y1, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (W, y1), (18, 18, 22), -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def hud(frame, prompt, attempt, n_attempts, lift_cm, status, color, rollouts):
    f = frame.copy()
    _bar(f, 0, 60)
    cv2.putText(f, "PHASE 2  -  LEARNED GRASP POLICY (Pi-0-style VLA)", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(f, f'task: "{prompt}"', (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 200, 230), 1, cv2.LINE_AA)
    _bar(f, H - 56, H)
    cv2.putText(f, f"part lift: {lift_cm:4.1f} cm", (12, H - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(f, f"attempt {attempt}/{n_attempts}   policy tuned over {rollouts:,} sim rollouts",
                (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 190), 1, cv2.LINE_AA)
    # status chip
    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(f, (W - tw - 26, H - 44), (W - 10, H - 18), color, -1)
    cv2.putText(f, status, (W - tw - 18, H - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return f


def render_attempts(env, theta, seeds, prompt, rollouts):
    """Run the policy on each seed, return a flat list of HUD frames + per-attempt
    success flags."""
    all_frames, successes = [], []
    n = len(seeds)
    for i, s in enumerate(seeds):
        env.reset(seed=int(s))
        out = env.rollout(theta, record=True, frame_every=6)
        frames, heights, z0 = out["frames"], out["heights"], out["z0"]
        successes.append(out["success"])
        nf = len(frames)
        for k, (fr, h) in enumerate(zip(frames, heights)):
            lift = h - z0
            lift_cm = max(0.0, lift * 100)
            if lift > LIFT_SUCCESS:
                status, color = "LIFTED", (90, 200, 90)
            elif k < nf * 0.45:
                status, color = "REACHING", (60, 170, 230)
            else:
                status, color = "GRASPING", (60, 200, 230)
            # once the part is clearly up, lock the success chip on
            if out["success"] and k > nf * 0.6:
                status, color = "SUCCESS", (90, 210, 110)
            all_frames.append(hud(fr, prompt, i + 1, n, lift_cm, status, color, rollouts))
        # brief freeze on the last frame so each attempt reads clearly
        all_frames.extend([all_frames[-1]] * 8)
    return all_frames, successes


def write_mp4(path, frames, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def make_two_panel(frames, curve_png, out_path, fps=30):
    """Headline video: static learning curve on the left, live grasp on the right."""
    if not os.path.exists(curve_png):
        print(f"  (no learning curve at {curve_png}; skipping 2-panel demo)")
        return
    curve = cv2.imread(curve_png)
    scale = H / curve.shape[0]
    curve = cv2.resize(curve, (int(curve.shape[1] * scale), H))
    cw = curve.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (cw + W, H))
    for f in frames:
        vw.write(np.hstack([curve, f]))
    vw.release()
    print(f"saved headline 2-panel demo -> {out_path}  ({cw + W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="data/grasp_policy.npz")
    ap.add_argument("--curve", default="data/learning_curve.png")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=100, help="base seed for test placements")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--frames-out", default="",
                    help="optional pickle of raw frames (large; off by default). "
                         "Set e.g. data/sim_frames.pkl for the dashboard center panel.")
    ap.add_argument("--mp4", default="data/grasp_sim.mp4")
    ap.add_argument("--demo", default="data/PHASE2_DEMO.mp4")
    args = ap.parse_args()

    if os.path.exists(args.policy):
        d = np.load(args.policy, allow_pickle=True)
        theta = d["best_theta"]
        rollouts = int(d["total_rollouts"]) if "total_rollouts" in d else 0
        fin = float(d["final_success"]) if "final_success" in d else float("nan")
        print(f"loaded learned policy from {args.policy} "
              f"(trained over {rollouts:,} rollouts, held-out success {fin*100:.0f}%)")
    else:
        theta = PARAM_INIT
        rollouts = 0
        print(f"[warn] {args.policy} not found; using untrained init policy. "
              f"Run sim/train_grasp.py first.")
    print("policy:", {k: round(v, 3) for k, v in describe_theta(theta).items()})

    env = GraspEnv()
    seeds = [args.seed + i * 17 for i in range(args.attempts)]
    frames, successes = render_attempts(env, theta, seeds, args.prompt, rollouts)
    env.close()
    print(f"rendered {len(frames)} frames over {args.attempts} test placements "
          f"| grasp success: {sum(successes)}/{len(successes)}")

    os.makedirs(os.path.dirname(args.mp4) or ".", exist_ok=True)
    if args.frames_out:
        with open(args.frames_out, "wb") as f:
            pickle.dump({"frames": frames}, f)
        print(f"saved frames -> {args.frames_out}")
    write_mp4(args.mp4, frames)
    print(f"saved grasp video -> {args.mp4}")
    make_two_panel(frames, args.curve, args.demo)


if __name__ == "__main__":
    main()
