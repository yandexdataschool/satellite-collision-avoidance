# Baseline description

Baseline model represents the grid search for a collision-avoidance maneuver. This model provides the maneuver that is collinear to the velocity vector. If enabled, the model provides a reverse maneuver to return to the initial orbit as well.

Also, it should be noticed that the maneuvers table obtained by Baseline model could be tuned by some another models (for example, through CE model).

<!--TODO: time of maneuver-->

Examples:

- [Tutorial](../../../examples/Notebooks/tutorials/Baseline_tutorial.ipynb)
- [The comparison of models on the generated sample of environments](../../../examples/Notebooks/analysis_and_experiments/Models_comparison.ipynb)