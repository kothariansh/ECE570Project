# used some llms to help refine the code, make it more readable, help with syntax, etc. but the core logic and structure
# implemented by me
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_objective_bar(metrics, path):
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel("Average Weighted Objective")
    plt.title("Routing Policy Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_lambda_sweep(sweep_results, path):
    xs = [round(x["lambda_risk"], 2) for x in sweep_results]
    ys = [x["avg_objective"] for x in sweep_results]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Risk Weight")
    plt.ylabel("Average Weighted Objective")
    plt.title("Effect of Risk Weight on Goal-Conditioned Routing")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_sample_routes(coords, routes_dict, path):
    plt.figure(figsize=(12, 4))
    keys = list(routes_dict.keys())
    for i, name in enumerate(keys, start=1):
        plt.subplot(1, len(keys), i)
        route = routes_dict[name]
        full = [0] + route + [0]
        pts = coords[full]
        plt.scatter(coords[:, 0], coords[:, 1], s=25)
        plt.plot(pts[:, 0], pts[:, 1], marker="o")
        plt.scatter(coords[0, 0], coords[0, 1], s=80, marker="s")
        plt.title(name)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
