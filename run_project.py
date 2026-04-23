# used some llms to help refine the code, make it more readable, help with syntax, etc. but the core logic and structure
# implemented by me

import argparse
import os
import numpy as np

from routing import (
    compute_rtg,
    generate_instance,
    random_route,
    nearest_neighbor_route,
    goal_conditioned_greedy_route,
    route_objective,
    route_breakdown,
)
from analysis import save_json, plot_objective_bar, plot_lambda_sweep, plot_sample_routes


def evaluate_policies(num_instances, graph_size, seed):
    rng = np.random.default_rng(seed)
    lambdas = np.array([0.9, 0.05, 0.05], dtype=float)

    random_scores = []
    nearest_scores = []
    goal_scores = []
    sample = None

    for idx in range(num_instances):
        coords, traffic, risk = generate_instance(graph_size, rng)

        r1 = random_route(graph_size, rng)
        r2 = nearest_neighbor_route(coords)
        r3 = goal_conditioned_greedy_route(coords, traffic, risk, lambdas)

        random_scores.append(route_objective(r1, coords, traffic, risk, lambdas))
        nearest_scores.append(route_objective(r2, coords, traffic, risk, lambdas))
        goal_scores.append(route_objective(r3, coords, traffic, risk, lambdas))

        if sample is None:
            sample = {
                "coords": coords,
                "routes": {
                    "Random": r1,
                    "Nearest": r2,
                    "Goal-Conditioned": r3,
                },
                "breakdown": {
                    "Random": route_breakdown(r1, coords, traffic, risk),
                    "Nearest": route_breakdown(r2, coords, traffic, risk),
                    "Goal-Conditioned": route_breakdown(r3, coords, traffic, risk),
                },
            }

    return {
        "random_baseline": float(np.mean(random_scores)),
        "nearest_neighbor": float(np.mean(nearest_scores)),
        "goal_conditioned_greedy": float(np.mean(goal_scores)),
        "lambdas": lambdas.tolist(),
    }, sample


def run_lambda_sweep(num_instances, graph_size, seed):
    rng = np.random.default_rng(seed)
    risk_weights = np.linspace(0.0, 0.8, 9)
    sweep_results = []

    for rw in risk_weights:
        lambdas = np.array([0.6 - rw / 2.0, 0.4 - rw / 2.0, rw], dtype=float)
        lambdas = np.clip(lambdas, 0.05, None)
        lambdas = lambdas / lambdas.sum()
        scores = []

        for _ in range(num_instances):
            coords, traffic, risk = generate_instance(graph_size, rng)
            route = goal_conditioned_greedy_route(coords, traffic, risk, lambdas)
            scores.append(route_objective(route, coords, traffic, risk, lambdas))

        sweep_results.append({
            "lambda_distance": float(lambdas[0]),
            "lambda_traffic": float(lambdas[1]),
            "lambda_risk": float(lambdas[2]),
            "avg_objective": float(np.mean(scores)),
        })

    return sweep_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics, sample = evaluate_policies(args.num_instances, args.graph_size, args.seed)
    sweep = run_lambda_sweep(max(40, args.num_instances // 2), args.graph_size, args.seed + 1)

    # RTG-style value from the sample goal-conditioned route
    goal_breakdown = sample["breakdown"]["Goal-Conditioned"]
    example_rtg = compute_rtg(
        remaining_time=goal_breakdown["distance"],
        remaining_cost=goal_breakdown["traffic"],
        risk=goal_breakdown["risk"],
        lambdas=np.array(metrics["lambdas"]),
    )

    results = {
        "settings": {
            "num_instances": args.num_instances,
            "graph_size": args.graph_size,
            "seed": args.seed,
        },
        "metrics": metrics,
        "sample_route_breakdown": sample["breakdown"],
        "example_rtg_value": float(example_rtg),
        "lambda_sweep": sweep,
    }

    save_json(results, os.path.join(args.output_dir, "results.json"))
    plot_objective_bar(
        {
            "Random": metrics["random_baseline"],
            "Nearest": metrics["nearest_neighbor"],
            "Goal-Conditioned": metrics["goal_conditioned_greedy"],
        },
        os.path.join(args.output_dir, "average_objective_comparison.png"),
    )
    plot_lambda_sweep(sweep, os.path.join(args.output_dir, "lambda_sweep.png"))
    plot_sample_routes(
        sample["coords"],
        sample["routes"],
        os.path.join(args.output_dir, "sample_routes.png"),
    )

    print("Finished.")
    print("Average weighted objective:")
    print(f"  Random baseline      : {metrics['random_baseline']:.4f}")
    print(f"  Nearest-neighbor     : {metrics['nearest_neighbor']:.4f}")
    print(f"  Goal-conditioned     : {metrics['goal_conditioned_greedy']:.4f}")
    print(f"  Example RTG value    : {results['example_rtg_value']:.4f}")
    print(f"Saved outputs to: {args.output_dir}/")


if __name__ == "__main__":
    main()
