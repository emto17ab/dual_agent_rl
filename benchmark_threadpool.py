"""
Benchmark: Reusable ThreadPoolExecutor vs. creating a new one each step vs. sequential.

Tests whether threading overhead is worth it for solveRebFlow calls.

Usage:
  bsub < benchmark_threadpool.sh
  # or interactively:
  python benchmark_threadpool.py --city nyc_man_south --mode 2 --num_episodes 3
"""

import argparse
import time
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.algos.reb_flow_solver_multi_agent import solveRebFlow
from src.envs.amod_env_multi import Scenario, AMoD
from src.misc.utils import dictsum

# --- Config ---
demand_ratio = {'san_francisco': 2, 'nyc_man_south': 1.0, 'washington_dc': 4.2}
json_hr = {'san_francisco': 19, 'nyc_man_south': 19, 'washington_dc': 19}
beta = {'san_francisco': 0.2, 'nyc_man_south': 0.5, 'washington_dc': 0.5}
choice_intercept = {'san_francisco': 14.15, 'nyc_man_south': 9.84, 'washington_dc': 11.75}
wage = {'san_francisco': 17.76, 'nyc_man_south': 22.77, 'washington_dc': 25.26}

parser = argparse.ArgumentParser(description="Benchmark ThreadPoolExecutor for solveRebFlow")
parser.add_argument("--city", type=str, default="nyc_man_south")
parser.add_argument("--mode", type=int, default=2, choices=[0, 2, 4],
                    help="Only modes 0, 2, 4 use solveRebFlow")
parser.add_argument("--max_steps", type=int, default=20)
parser.add_argument("--num_episodes", type=int, default=3, help="Episodes per benchmark")
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--supply_ratio", type=float, default=1.0)
parser.add_argument("--agent0_vehicle_ratio", type=float, default=0.5)
args = parser.parse_args()

city = args.city

if args.mode not in [0, 2, 4]:
    print(f"Mode {args.mode} does not use solveRebFlow. Nothing to benchmark.")
    print("Use --mode 0, 2, or 4.")
    sys.exit(0)


def build_uniform_desired_acc(env, agent_id):
    """Build a uniform desiredAcc (mimics fixed agent / mode 4 logic)."""
    current_total = dictsum(env.agent_acc[agent_id], env.time + 1)
    if current_total == 0:
        return {env.region[i]: 0 for i in range(env.nregion)}
    base_per_region = current_total // env.nregion
    remainder = current_total % env.nregion
    return {
        env.region[i]: base_per_region + (1 if i < remainder else 0)
        for i in range(env.nregion)
    }


def run_episode(env, mode, reb_solver_fn):
    """
    Run one full episode with fixed prices and the given reb_solver_fn.
    reb_solver_fn(env, desiredAcc_dict) -> rebAction dict {0: [...], 1: [...]}

    Returns (elapsed_seconds, num_reb_calls, reb_time_total).
    """
    env.reset()
    done = False
    reb_calls = 0
    reb_time_total = 0.0

    # Fixed price action (0.5 for all regions)
    if mode == 0:
        price_action = None  # mode 0 match_step_simple takes no price arg
    elif mode == 2:
        # Mode 2 expects {agent: array of shape [nregion, 2]} for price+reb
        # Use 0.5 price, uniform reb proportion
        price_action = {}
        for a in [0, 1]:
            total_v = sum(env.agent_initial_acc[a].values())
            reb_props = np.array([
                env.agent_initial_acc[a][env.region[i]] / max(total_v, 1)
                for i in range(env.nregion)
            ])
            price_action[a] = np.column_stack([
                np.full(env.nregion, 0.5),
                reb_props
            ])
    else:  # mode 4
        price_action = {
            a: np.full(env.nregion, 0.5) for a in [0, 1]
        }

    t_start = time.perf_counter()

    while not done:
        # Match step
        if mode == 0:
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
        else:
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(price_action)

        if done:
            break

        # Build desiredAcc
        desiredAcc = {a: build_uniform_desired_acc(env, a) for a in [0, 1]}

        # Solve rebalancing (timed separately)
        t_reb = time.perf_counter()
        rebAction = reb_solver_fn(env, desiredAcc)
        reb_time_total += time.perf_counter() - t_reb
        reb_calls += 1

        if rebAction[0] is None or rebAction[1] is None:
            print(f"  WARNING: solveRebFlow returned None at step {env.time}")
            break

        # Reb step
        _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)

    elapsed = time.perf_counter() - t_start
    return elapsed, reb_calls, reb_time_total


# --- Solver variants ---

def solve_new_pool(env, desiredAcc):
    """Create a NEW ThreadPoolExecutor each call."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda a: solveRebFlow(env, desiredAcc[a], a),
            [0, 1]
        ))
    return {0: results[0], 1: results[1]}


def make_reusable_solver(executor):
    """Return a solver that reuses the given ThreadPoolExecutor."""
    def solve(env, desiredAcc):
        results = list(executor.map(
            lambda a: solveRebFlow(env, desiredAcc[a], a),
            [0, 1]
        ))
        return {0: results[0], 1: results[1]}
    return solve


def solve_sequential(env, desiredAcc):
    """Solve rebalancing sequentially (no threading)."""
    r0 = solveRebFlow(env, desiredAcc[0], 0)
    r1 = solveRebFlow(env, desiredAcc[1], 1)
    return {0: r0, 1: r1}


# --- Setup ---
print("=" * 60)
print("ThreadPoolExecutor Benchmark for solveRebFlow")
print("=" * 60)
print(f"City:       {city}")
print(f"Mode:       {args.mode}")
print(f"Max steps:  {args.max_steps}")
print(f"Episodes:   {args.num_episodes}")
print(f"Seed:       {args.seed}")
print()

print("Setting up environment...", flush=True)
scenario = Scenario(
    json_file=f"data/scenario_{city}.json",
    demand_ratio=demand_ratio[city] * 2,
    json_hr=json_hr[city],
    sd=args.seed,
    json_tstep=3,
    tf=args.max_steps,
    impute=0,
    supply_ratio=args.supply_ratio,
    agent0_vehicle_ratio=args.agent0_vehicle_ratio,
    total_vehicles=None,
)

env = AMoD(
    scenario, args.mode, beta=beta[city], jitter=1, max_wait=2,
    choice_price_mult=1.0, seed=args.seed, fix_agent=2,
    choice_intercept=choice_intercept[city], wage=wage[city],
    use_dynamic_wage_man_south=False, od_price_actions=False,
)

print(f"Regions:    {env.nregion}")
print(f"Edges:      {len(env.edges)}")
print()

# --- Warmup (1 episode to prime CPLEX, caches, etc.) ---
print("Warmup run (1 episode, sequential)...", flush=True)
elapsed, calls, reb_time = run_episode(env, args.mode, solve_sequential)
print(f"  Warmup done: {elapsed:.4f}s total, {calls} reb calls, {reb_time:.4f}s in solveRebFlow")
if calls > 0:
    print(f"  ~{reb_time/calls*1000:.2f}ms per solveRebFlow call pair (2 agents sequential)")
print()

# --- Benchmarks ---
benchmarks = [
    ("1) New ThreadPool each step", solve_new_pool),
]

reusable_executor = ThreadPoolExecutor(max_workers=2)
benchmarks.append(("2) Reusable ThreadPool", make_reusable_solver(reusable_executor)))
benchmarks.append(("3) Sequential (no threading)", solve_sequential))

results = {}
for name, solver in benchmarks:
    print(f"{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    times = []
    reb_times = []
    total_calls = 0
    for ep in range(args.num_episodes):
        elapsed, calls, reb_time = run_episode(env, args.mode, solver)
        times.append(elapsed)
        reb_times.append(reb_time)
        total_calls += calls
        print(f"  Episode {ep+1}/{args.num_episodes}: "
              f"{elapsed:.4f}s total, {reb_time:.4f}s in reb ({calls} calls)")

    avg_total = np.mean(times)
    std_total = np.std(times)
    avg_reb = np.mean(reb_times)
    results[name] = {
        "avg_total": avg_total, "std_total": std_total,
        "avg_reb": avg_reb, "total_calls": total_calls
    }
    print(f"  Avg total: {avg_total:.4f}s +/- {std_total:.4f}s")
    print(f"  Avg reb:   {avg_reb:.4f}s")
    print()

reusable_executor.shutdown(wait=True)

# --- Summary ---
print(f"{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name:40s}: {r['avg_total']:.4f}s +/- {r['std_total']:.4f}s "
          f"(reb: {r['avg_reb']:.4f}s, {r['total_calls']} calls)")
print()

names = list(results.keys())
new_r = results[names[0]]
reuse_r = results[names[1]]
seq_r = results[names[2]]

print("Pairwise comparisons (total episode time):")
d1 = (new_r['avg_total'] - reuse_r['avg_total']) / new_r['avg_total'] * 100
print(f"  Reusable vs New pool:  {'+' if d1>0 else ''}{d1:.1f}% "
      f"({'faster' if d1>0 else 'slower'})")

d2 = (new_r['avg_total'] - seq_r['avg_total']) / new_r['avg_total'] * 100
print(f"  Sequential vs New pool: {'+' if d2>0 else ''}{d2:.1f}% "
      f"({'faster' if d2>0 else 'slower'})")

d3 = (reuse_r['avg_total'] - seq_r['avg_total']) / reuse_r['avg_total'] * 100
print(f"  Sequential vs Reusable: {'+' if d3>0 else ''}{d3:.1f}% "
      f"({'faster' if d3>0 else 'slower'})")

print()
print("Pairwise comparisons (reb solver time only):")
d4 = (new_r['avg_reb'] - reuse_r['avg_reb']) / max(new_r['avg_reb'], 1e-9) * 100
print(f"  Reusable vs New pool:  {'+' if d4>0 else ''}{d4:.1f}%")
d5 = (new_r['avg_reb'] - seq_r['avg_reb']) / max(new_r['avg_reb'], 1e-9) * 100
print(f"  Sequential vs New pool: {'+' if d5>0 else ''}{d5:.1f}%")

print()
best_name = min(results, key=lambda k: results[k]["avg_total"])
worst_name = max(results, key=lambda k: results[k]["avg_total"])
best = results[best_name]["avg_total"]
worst = results[worst_name]["avg_total"]

print(f"FASTEST: {best_name} ({best:.4f}s)")
print(f"SLOWEST: {worst_name} ({worst:.4f}s)")
print()

# Recommendation
if seq_r['avg_total'] <= min(new_r['avg_total'], reuse_r['avg_total']):
    print(">>> RECOMMENDATION: Use SEQUENTIAL (no threading).")
    print("    solveRebFlow is too fast for threading overhead to pay off.")
    print("    Remove ThreadPoolExecutor from main_a2c_multi_agent.py")
elif reuse_r['avg_total'] < new_r['avg_total']:
    print(">>> RECOMMENDATION: Use REUSABLE ThreadPoolExecutor.")
    print("    Create one executor before the training loop and reuse it.")
else:
    print(">>> RECOMMENDATION: Current approach (new pool each step) is fine.")

# Extrapolate savings
episodes = 100000
savings_per_ep = worst - best
total_hours = savings_per_ep * episodes / 3600
print(f"\nEstimated savings over {episodes:,} episodes: {total_hours:.1f} hours")
print(f"  ({savings_per_ep*1000:.2f}ms per episode x {episodes:,} episodes)")
print(f"\n{'='*60}")
print("Done.")
