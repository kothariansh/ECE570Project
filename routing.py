# used some llms to help refine the code, make it more readable, help with syntax, etc. but the core logic and structure
# implemented by me
import numpy as np


def compute_rtg(remaining_time, remaining_cost, risk, lambdas):
    l1, l2, l3 = lambdas
    return -(l1 * remaining_time + l2 * remaining_cost + l3 * risk)


def generate_instance(graph_size, rng):
    # Create a synthetic routing instance.
    # Node 0 is the depot/start.
    coords = rng.random((graph_size + 1, 2))
    traffic = 0.8 + 0.9 * rng.random(graph_size + 1) # traffic multiplier
    risk = rng.random(graph_size + 1) # node risk score
    traffic[0] = 1.0
    risk[0] = 0.0
    return coords, traffic, risk


def step_components(curr, nxt, coords, traffic, risk):
    dist = float(np.linalg.norm(coords[curr] - coords[nxt]))
    traffic_cost = float(dist * traffic[nxt])
    risk_cost = float(risk[nxt])
    return dist, traffic_cost, risk_cost


def weighted_step_cost(curr, nxt, coords, traffic, risk, lambdas):
    l1, l2, l3 = lambdas
    dist, traffic_cost, risk_cost = step_components(curr, nxt, coords, traffic, risk)
    return l1 * dist + l2 * traffic_cost + l3 * risk_cost


def route_objective(route, coords, traffic, risk, lambdas):
    total = 0.0
    curr = 0
    for nxt in route:
        total += weighted_step_cost(curr, nxt, coords, traffic, risk, lambdas)
        curr = nxt
    # return to depot with distance component only
    total += lambdas[0] * float(np.linalg.norm(coords[curr] - coords[0]))
    return float(total)


def random_route(graph_size, rng):
    route = list(range(1, graph_size + 1))
    rng.shuffle(route)
    return route


def nearest_neighbor_route(coords):
    n = len(coords) - 1
    unvisited = set(range(1, n + 1))
    curr = 0
    route = []
    while unvisited:
        nxt = min(unvisited, key=lambda j: np.linalg.norm(coords[curr] - coords[j]))
        route.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return route


def goal_conditioned_greedy_route(coords, traffic, risk, lambdas):
    n = len(coords) - 1
    unvisited = set(range(1, n + 1))
    curr = 0
    route = []
    while unvisited:
        nxt = min(unvisited, key=lambda j: weighted_step_cost(curr, j, coords, traffic, risk, lambdas))
        route.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return route


def route_breakdown(route, coords, traffic, risk):
    dist_sum = 0.0
    traffic_sum = 0.0
    risk_sum = 0.0
    curr = 0
    for nxt in route:
        d, t, r = step_components(curr, nxt, coords, traffic, risk)
        dist_sum += d
        traffic_sum += t
        risk_sum += r
        curr = nxt
    return {
        "distance": float(dist_sum),
        "traffic": float(traffic_sum),
        "risk": float(risk_sum),
    }
