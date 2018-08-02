# CE description

Cross-Entropy (CE) model is based on the Reinforcement Learning Cross-Entropy method.
This model provides maneuvers table by some inital table through an iterative shifting.
The approximate algorithm of such iteration is the following:

1. __generate__ a sample of random maneuvers tables from the initial table;
2. __evaluate__ each maneuvers table by the reward at the end of the session;
3. __select__ some tables with the best reward (using a percentile);
4. __shift__ the initial table in the direction of the best matrices (an average matrix).

This model has a lot of options (that could be changed during the model training, by the way) and possibilities:

- to set the first maneuver time - could be "early" or "auto";
- to set the maneuvers angle relative to the velocity vector - could be "complanar", "collinear" or "auto";
- to add a reverse maneuver to return to the initial orbit.

CE model has a number of not always pleasant features as well:

- user-defined maneuvers number;
- a strong dependence on the initial maneuvers table;
- a plenty of hyperparameters;
- a great chance to get into the local optimum.

Also, it should be noticed that CE model could be used for tuning maneuvers table obtained by another models.

<!--TODO: time of maneuver-->

Examples:

- [CE tutorial](../../../examples/Notebooks/tutorials/CE_tutorial.ipynb)
- [The comparison of models on the generated sample of environments](../../../examples/Notebooks/analysis_and_experiments/Models_comparison.ipynb)