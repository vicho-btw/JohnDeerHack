"""
sim/verify_tasks.py — verify each Phase 2 task over many random seeds.

For every task, runs N randomized rollouts in its shipping mode (real grasp or
weld-assist) and reports the physics-based success rate. With --both it also
probes the opposite grip mode so you can see why weld-assist was chosen. Results
are appended to STATUS.md.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.manip_env import ManipEnv
from sim.tasks import TASKS, TASK_ORDER, run_task

FR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "franka")


def verify(name, n=30, weld=None):
    task = TASKS[name]
    env = ManipEnv(os.path.join(FR, f"task_{task.scene}.xml"))
    ok = 0
    zs = []
    t0 = time.time()
    for s in range(n):
        success, z, text, _ = run_task(env, task, seed=1000 + s, weld=weld)
        ok += int(bool(success))
        zs.append(z)
    env.close()
    dt = time.time() - t0
    return {"name": name, "succ": ok / n, "n": n, "dt": dt,
            "mode": "weld" if (task.default_weld if weld is None else weld) else "real"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tasks", nargs="*", default=TASK_ORDER)
    ap.add_argument("--log", default="STATUS.md")
    args = ap.parse_args()

    lines = []
    print(f"verifying {len(args.tasks)} tasks x {args.n} random seeds\n")
    for name in args.tasks:
        r = verify(name, n=args.n)
        msg = (f"{name:11s} mode={r['mode']:4s} success={r['succ']*100:5.1f}% "
               f"({int(r['succ']*r['n'])}/{r['n']})  [{r['dt']:.1f}s]")
        print(msg)
        lines.append(msg)

    if args.log:
        with open(args.log, "a") as f:
            f.write(f"\n### verify_tasks ({args.n} seeds/task)\n\n```\n")
            f.write("\n".join(lines))
            f.write("\n```\n")
    print(f"\nappended results to {args.log}")


if __name__ == "__main__":
    main()
