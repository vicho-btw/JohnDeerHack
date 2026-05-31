"""
sim/train_grasp.py — PHASE 2 training: learn the grasp policy in simulation.

The "policy" is the parameter vector `theta` consumed by sim/grasp_env.py. We
optimize it with the Cross-Entropy Method (CEM): sample a population of policies
from a Gaussian, run each through REAL MuJoCo grasp rollouts on randomized part
placements, keep the best ("elite") few, refit the Gaussian to them, repeat.

This is a genuine learning loop: it runs thousands of physics rollouts and the
success rate climbs from near-zero (random policies almost always drop the part)
to near-perfect. The honest pitch claim — "the policy was tuned over thousands of
simulated grasp attempts" — is exactly what this does.

Outputs:
    data/grasp_policy.npz   learned theta + full training history
    data/learning_curve.png success-rate / reward vs. CEM iteration

Usage:
    python sim/train_grasp.py                       # default ~5-6k rollouts
    python sim/train_grasp.py --iters 40 --pop 40   # bigger run
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.grasp_env import (GraspEnv, evaluate, PARAM_LOW, PARAM_HIGH, N_PARAMS,
                           PARAM_NAMES, clip_theta)


def cem(iters=30, pop=32, elite_frac=0.25, train_k=6, eval_k=16, seed=0,
        verbose=True):
    """Run CEM. Returns (best_theta, mean_theta, history dict)."""
    rng = np.random.default_rng(seed)
    env = GraspEnv()  # one env reused for every rollout (fast, no reload)

    # Broad initial Gaussian centered in the parameter box -> early policies are
    # effectively random and mostly fail, so the learning curve is meaningful.
    mean = 0.5 * (PARAM_LOW + PARAM_HIGH)
    std = 0.5 * (PARAM_HIGH - PARAM_LOW)
    std_floor = 0.02 * (PARAM_HIGH - PARAM_LOW)
    n_elite = max(2, int(round(pop * elite_frac)))

    hist = {"iter": [], "mean_success": [], "pop_success": [], "best_fitness": [],
            "pop_fitness": [], "rollouts": []}
    best_theta, best_fit = clip_theta(mean), -1e9
    total_rollouts = 0
    t0 = time.time()

    for it in range(iters):
        # shared placement seeds this iteration -> fair comparison across policies
        train_seeds = rng.integers(0, 1_000_000, size=train_k)
        samples = rng.normal(mean, std, size=(pop, N_PARAMS))
        samples = np.clip(samples, PARAM_LOW, PARAM_HIGH)

        fitness = np.empty(pop)
        succ = np.empty(pop)
        for i in range(pop):
            stats = evaluate(samples[i], train_seeds, env)
            fitness[i] = stats["reward"]
            succ[i] = stats["success_rate"]
        total_rollouts += pop * train_k
        pop_success = float(succ.mean())  # fraction of sampled grasps that lifted

        order = np.argsort(fitness)[::-1]
        elite = samples[order[:n_elite]]
        mean = 0.7 * elite.mean(axis=0) + 0.3 * mean      # smoothed update
        std = np.maximum(0.7 * elite.std(axis=0) + 0.3 * std, std_floor)

        if fitness[order[0]] > best_fit:
            best_fit = float(fitness[order[0]])
            best_theta = clip_theta(samples[order[0]])

        # honest progress metric: success rate of the CURRENT mean policy on a
        # fresh held-out set of placements (not used for selection)
        eval_seeds = rng.integers(0, 1_000_000, size=eval_k)
        mean_stats = evaluate(clip_theta(mean), eval_seeds, env)
        total_rollouts += eval_k

        hist["iter"].append(it)
        hist["mean_success"].append(mean_stats["success_rate"])
        hist["pop_success"].append(pop_success)
        hist["best_fitness"].append(best_fit)
        hist["pop_fitness"].append(float(fitness.mean()))
        hist["rollouts"].append(total_rollouts)

        if verbose:
            print(f"iter {it:2d} | population grasp success {pop_success*100:5.1f}% "
                  f"| mean-policy {mean_stats['success_rate']*100:5.1f}% "
                  f"| pop reward {fitness.mean():+.3f} | rollouts {total_rollouts:5d} "
                  f"| {time.time()-t0:5.1f}s", flush=True)

    # final eval of the best policy over a large held-out set
    final_seeds = rng.integers(0, 1_000_000, size=64)
    final = evaluate(best_theta, final_seeds, env)
    total_rollouts += 64
    env.close()
    hist["total_rollouts"] = total_rollouts
    hist["final"] = final
    return best_theta, clip_theta(mean), hist


def plot_curve(hist, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.2), dpi=120)
    it = hist["iter"]
    ax1.plot(it, np.array(hist["pop_success"]) * 100, "-o", color="#1f8a4c",
             lw=2.4, ms=4, label="sampled-policy grasp success")
    ax1.plot(it, np.array(hist["mean_success"]) * 100, "-", color="#2f6fb0",
             lw=1.6, alpha=0.8, label="best-estimate policy (held-out)")
    ax1.set_xlabel("CEM iteration")
    ax1.set_ylabel("grasp success rate (%)", color="#1f8a4c")
    ax1.set_ylim(-3, 103)
    ax1.tick_params(axis="y", labelcolor="#1f8a4c")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(it, hist["pop_fitness"], "--", color="#b0602a", lw=1.6,
             label="population mean reward")
    ax2.set_ylabel("reward", color="#b0602a")
    ax2.tick_params(axis="y", labelcolor="#b0602a")
    n = hist.get("total_rollouts", hist["rollouts"][-1])
    fin = hist["final"]["success_rate"] * 100
    ax1.set_title(f"Phase 2: grasp policy learned over {n:,} simulated rollouts\n"
                  f"final held-out success: {fin:.0f}%", fontsize=11)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--train-k", type=int, default=6, help="placements per policy eval")
    ap.add_argument("--eval-k", type=int, default=16, help="held-out placements")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/grasp_policy.npz")
    ap.add_argument("--curve", default="data/learning_curve.png")
    args = ap.parse_args()

    print(f"CEM: {args.iters} iters x {args.pop} pop x {args.train_k} placements "
          f"= {args.iters*args.pop*args.train_k} training rollouts (+ evals)\n")
    best, mean, hist = cem(iters=args.iters, pop=args.pop, elite_frac=0.25,
                           train_k=args.train_k, eval_k=args.eval_k, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, best_theta=best, mean_theta=mean,
             param_names=np.array(PARAM_NAMES),
             iter=np.array(hist["iter"]),
             mean_success=np.array(hist["mean_success"]),
             pop_success=np.array(hist["pop_success"]),
             best_fitness=np.array(hist["best_fitness"]),
             pop_fitness=np.array(hist["pop_fitness"]),
             rollouts=np.array(hist["rollouts"]),
             total_rollouts=hist["total_rollouts"],
             final_success=hist["final"]["success_rate"],
             final_lift=hist["final"]["mean_lift"])
    plot_curve(hist, args.curve)

    print(f"\nlearned policy (best of {hist['total_rollouts']:,} rollouts):")
    for name, v in zip(PARAM_NAMES, best):
        print(f"  {name:12s} {v:+.4f}")
    print(f"\nfinal held-out success: {hist['final']['success_rate']*100:.1f}% "
          f"over 64 random placements | mean lift {hist['final']['mean_lift']:.3f} m")
    print(f"saved policy -> {args.out}")
    print(f"saved learning curve -> {args.curve}")


if __name__ == "__main__":
    main()
