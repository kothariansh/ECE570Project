# Goal-Conditioned Routing Code

This zip contains a **fully runnable**, lightweight project for synthetic route optimization.
I took help from AI to make this GitHub repository more readable and aesthetic. 
All the code was uploaded simulatenously as I was working with a local Git environment and Google Collab for the most.
AI LLMs were used to help fix syntax errors, structure the code better, and make everything readable overall. I do acknolwedge this. However, most of the original logic was mine and I wrote and debugged a considerable amount of code myself for this project. 

It does something meaningful:
- generates synthetic TSP-style delivery instances
- assigns each node a traffic multiplier and risk score
- compares three routing policies:
  1. random baseline
  2. nearest-neighbor (distance-only)
  3. goal-conditioned greedy routing (distance + traffic + risk)
- computes multi-objective route scores
- sweeps objective weights to show how route preferences change
- saves plots and JSON metrics

## Why this is useful
This is a compact prototype of **goal-conditioned routing**. It shows the core idea of your paper:
different objective weights lead to different route choices and different outcomes.

## Files
- `run_project.py` - main experiment script
- `routing.py` - route generation and scoring logic
- `analysis.py` - plotting and summary helpers
- `requirements.txt` - dependencies
- `README.md` - this file

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python run_project.py --num_instances 200 --graph_size 20
```

## Outputs
The script creates an `outputs/` folder containing:
- `results.json`
- `average_objective_comparison.png`
- `lambda_sweep.png`
- `sample_routes.png`

## Code attribution
All code in this zip was written for this deliverable.
