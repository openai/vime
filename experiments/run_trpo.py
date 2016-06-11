import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(2)
# SwimmerGather hierarchical task
mdp_classes = [SwimmerGatherEnv]
mdps = [NormalizedEnv(env=mdp_class())
        for mdp_class in mdp_classes]
param_cart_product = itertools.product(
    mdps, seeds
)

for mdp, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 32),
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    batch_size = 50000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=10000,
        step_size=0.01,
        subsample_factor=1.0,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo",
        n_parallel=4,
        snapshot_mode="last",
        seed=seed,
        mode="local"
    )
