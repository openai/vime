# How to run VIME

Variational Information Maximizing Exploration (VIME) as presented in Curiosity-driven Exploration in Deep Reinforcement Learning via Bayesian Neural Networks by *R. Houthooft, X. Chen, Y. Duan, J. Schulman, F. De Turck, P. Abbeel* (http://arxiv.org/abs/1605.09674). 

To reproduce the results, you should first have [rllab](https://github.com/rllab/rllab) and Mujoco v1.31 configured. Then, run the following commands in the root folder of `rllab`:

```bash
git submodule add -f git@github.com:openai/vime.git sandbox/vime
touch sandbox/__init__.py
```

Then you can do the following:
- Execute TRPO+VIME on the hierarchical SwimmerGather environment via `python sandbox/vime/experiments/run_trpo_expl.py`.
