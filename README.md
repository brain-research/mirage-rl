# The Mirage of Action-Dependent Baselines in Reinforcement Learning

Code to reproduce the experiments in *The Mirage of Action-Dependent Baselines in Reinforcement Learning*. George Tucker, Surya Bhupatiraju, Shixiang Gu, Richard E. Turner, Zoubin Ghahramani, Sergey Levine. ICML 2018. (https://arxiv.org/abs/1802.10031)

## Linear-Quadratic-Gaussian (LQG) systems
See Appendix Section 9 for a detailed description of the LQG system. The code in
this folder was used to generate the results in the LQG section (3.1) and
Figures 1 and 5.

## Q-Prop (https://arxiv.org/abs/1611.02247)
We modified the Q-Prop implementation published
by the authors at https://github.com/shaneshixiang/rllabplusplus (commit: 4d55f96). For our experiments, we used the conservative variant of QProp,
as is used throughout the experimental section in the
original paper. We used the default choices of policy and
value functions, learning rates, and other hyperparameters. This code was
used to generate Figure 3 and we describe the modifications in detail in Appendix 8.1.

The experimental data for all the results is contained in `data/local/*`. To run the plotter to get the same results as in the paper, you can run `python plot_rewards.py` or you can run `python plot_rewards.py --mini` to generate the same plot where each subfigure has its own legend (useful for cropping).

NOTE: Running the experiments found in `sandbox/rocky/tf/launchers/sample_run.sh` might throw a `ModuleNotFoundError`. To fix this, add the top-level folder to your environment variable `PYTHONPATH`.

## Backpropagation through the Void (https://arxiv.org/abs/1711.00123)
We used the implementation published by the authors
(https://github.com/wgrathwohl/BackpropThroughTheVoidRL, commit: 0e6623d)
with the following modification: we measure the variance
of the policy gradient estimator. In the original code, the
authors accidentally measure the variance of a gradient
estimator that neither method uses. We note that Grathwohl
et al. (2018) recently corrected a bug in the code that caused
the LAX method to use a different advantage estimator than
the base method. We use this bug fix. The code was used to generate Figure 13.

To generate the figure, run the following commands:
```
baselines/a2c/run_a2c.sh
baselines/a2c/plot_a2c.py
```
## Action-depedent Control Variates for Policy Optimization via Stein's Identity (https://arxiv.org/abs/1710.11198)
We used the Stein control variate implementation published
by the authors at https://github.com/DartML/PPO-Stein-Control-Variate (commit: 6eec471). We describe the experiments in
Appendix Section 8.2 and use the code to generate Figures 8 and 12.

To generate Figure 8, first create runner scripts with
```
python create_run_scripts.py
```
Then run the bash scripts to generate results. Use
```
python plot_gcp.py
```
to generate the figure from the log files (included in the repo).

To generate Figure 12,
```
bash walker2d_train_eval.sh
python traj_visualize.py
```
## TRPO experiments
We modified the open-source
TRPO implementation: https://github.com/ikostrikov/pytorch-trpo (commit: 27400b8).

### Performance comparison
To generate the performance comparison plot (Figure 4), switch to branch state_comparison and run the commands in the run_*.sh scripts and copy down the logs. Then run
plot.py to generate Figure 4.

### Variance calculations
To generate the variance plots (Figures 2, 9, 10, and 11), switch to branch
variance and run
```
run_train_models_for_variance.sh
run_calc_variance.sh
run_plot_variances.sh
```

### Horizon-aware Comparison
To generate the figures for the horizon-aware comparison experiments (Figures 6 and 7), switch to branch horizon_aware_comparison and you will need to run the training done in:
```
bash run_horizon_time_comparison.sh
```
This script uses a simple utility (berg, not included) to schedule jobs on
Google Compute Platform. The resulting log files are included in the gs_results
folder.

To generate the figure from the data, run
```
python plot_disc.py
```

This is not an officially supported Google product. George Tucker
(gjt@google.com) maintains this.
