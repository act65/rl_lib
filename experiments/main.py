"""Run script for reinforcement learning experiments.

This script sets up and runs RL experiments, configurable via command-line flags.
It allows specifying the environment, agent type, accumulator, and various
hyperparameters for training and evaluation.
"""

from absl import app
from absl import flags

# BSuite environments
from bsuite.environments import catch, cartpole, mountain_car, bandit

import dm_env  # For environment type hinting
import haiku as hk
import jax  # For jax.Array type hint
import tensorflow as tf  # For tf.TensorSpec
from typing import Any, Dict

from rl_lib.utils import NN, build_signature
from rl_lib.experiment import run_loop
from rl_lib import learners as learners_lib
from rl_lib import accumulators as accumulators_lib

# Import all agent and accumulator classes that will be selectable via flags
from rl_lib.learners import QLearner, SoftWatkins, EMATargetNet
from rl_lib.accumulators import (
    OnlineAccumulator,
    MultiStepOnlineAccumulator,
    ReverbAccumulator,
)


FLAGS = flags.FLAGS
# General flags
flags.DEFINE_integer("seed", 42, "Random seed for reproducibility.")
flags.DEFINE_integer("train_episodes", 1000, "Total number of episodes for training.")
flags.DEFINE_integer(
    "eval_episodes", 100, "Number of episodes to run for each evaluation phase."
)
flags.DEFINE_integer(
    "evaluate_every", 50, "Frequency (in training episodes) at which to run evaluation."
)

# Environment flag
flags.DEFINE_enum(
    "environment_name",
    "catch",
    ["catch", "cartpole", "mountain_car", "bandit"],
    "Name of the BSuite environment to run.",
)

# Network flags
flags.DEFINE_integer(
    "num_hidden_units", 50, "Number of hidden units in the neural network."
)

# Agent flags
flags.DEFINE_enum(
    "agent_name",
    "softwatkins",
    ["softwatkins", "qlearner", "ematargetnet"],
    "Name of the agent to use. `ematargetnet` is SoftWatkins with EMA target network.",
)
flags.DEFINE_float(
    "kappa",
    0.01,
    "Kappa parameter for Soft Watkins Q-learning, controlling tolerance for off-policy corrections.",
)
flags.DEFINE_float(
    "_lambda",
    1.0,
    "Lambda parameter for Q-lambda or Soft Watkins, determining n-step bootstrapping.",
)
flags.DEFINE_float(
    "epsilon",
    0.01,
    "Epsilon value for epsilon-greedy exploration (if applicable to the agent).",
)  # Currently unused by default agents
flags.DEFINE_float(
    "discount_factor", 0.99, "Discount factor (gamma) for future rewards."
)
flags.DEFINE_float("learning_rate", 0.005, "Learning rate for the optimizer.")
flags.DEFINE_float(
    "ema_decay", 0.99, "EMA decay rate for the target network in EMATargetNet."
)


# Accumulator flags
flags.DEFINE_enum(
    "accumulator_name",
    "reverb",
    ["reverb", "online", "multistep_online"],
    "Name of the accumulator to use.",
)
flags.DEFINE_integer(
    "batch_size", 1, "Batch size for training updates."
)  # Used by Reverb and learner
flags.DEFINE_integer(
    "n_multistep",
    1,
    "Number of steps for N-step returns or sequence length for Reverb/MultiStepOnline accumulators.",
)
flags.DEFINE_integer(
    "reverb_port", 5000, "Port for Reverb server, if ReverbAccumulator is used."
)
# Consider adding flags for reverb_max_size and reverb_shuffle_buffer_size if more control is needed


def main(unused_arg: Any) -> None:
    """Sets up and runs the reinforcement learning experiment based on flag configurations.

    This involves:
    1.  Initializing the specified environment.
    2.  Creating the neural network.
    3.  Instantiating the specified agent.
    4.  Instantiating the specified accumulator.
    5.  Executing the main training and evaluation loop.

    Args:
        unused_arg: Unused command-line arguments passed by `app.run`.
    """
    # Environment setup
    env: dm_env.Environment
    if FLAGS.environment_name == "catch":
        env = catch.Catch(seed=FLAGS.seed)
    elif FLAGS.environment_name == "cartpole":
        env = cartpole.Cartpole(seed=FLAGS.seed)
    elif FLAGS.environment_name == "mountain_car":
        env = mountain_car.MountainCar(seed=FLAGS.seed)
    elif FLAGS.environment_name == "bandit":
        # Assuming SimpleBandit does not take a seed, or adjust as necessary
        env = bandit.SimpleBandit()
    else:
        raise ValueError(f"Unknown environment: {FLAGS.environment_name}")

    # Network setup
    def net_fn(obs: jax.Array) -> jax.Array:
        nn_module = NN(
            width=FLAGS.num_hidden_units,
            n_dense=2,
            output_dims=env.action_spec().num_values,
        )
        return nn_module(obs)

    network_transformed: hk.Transformed = hk.without_apply_rng(hk.transform(net_fn))

    # Agent setup
    agent: learners_lib.RLAgent
    if FLAGS.agent_name == "qlearner":
        agent = QLearner(
            network=network_transformed,
            observation_spec=env.observation_spec(),
            learning_rate=FLAGS.learning_rate,
        )
    elif FLAGS.agent_name == "softwatkins":
        agent = SoftWatkins(
            network=network_transformed,
            observation_spec=env.observation_spec(),
            learning_rate=FLAGS.learning_rate,
            kappa=FLAGS.kappa,
            _lambda=FLAGS._lambda,
        )
    elif FLAGS.agent_name == "ematargetnet":
        agent = EMATargetNet(
            network=network_transformed,
            observation_spec=env.observation_spec(),
            learning_rate=FLAGS.learning_rate,
            kappa=FLAGS.kappa,
            _lambda=FLAGS._lambda,
            ema_decay=FLAGS.ema_decay,
        )
    else:
        raise ValueError(f"Unknown agent: {FLAGS.agent_name}")

    # Accumulator setup
    accumulator: accumulators_lib.Accumulator
    if FLAGS.accumulator_name == "online":
        accumulator = OnlineAccumulator()
    elif FLAGS.accumulator_name == "multistep_online":
        accumulator = MultiStepOnlineAccumulator(n_multistep=FLAGS.n_multistep)
    elif FLAGS.accumulator_name == "reverb":
        signature: Dict[str, tf.TensorSpec] = build_signature(
            env.observation_spec(), env.action_spec(), FLAGS.n_multistep
        )
        accumulator = ReverbAccumulator(
            port=FLAGS.reverb_port,
            signature=signature,
            max_size=1000,  # Default, consider making a flag
            n_multistep=FLAGS.n_multistep,
            batch_size=FLAGS.batch_size,
            shuffle_buffer_size=1000,  # Default, consider making a flag
        )
    else:
        raise ValueError(f"Unknown accumulator: {FLAGS.accumulator_name}")

    # Run the experiment loop
    run_loop(
        agent=agent,
        environment=env,
        accumulator=accumulator,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        train_episodes=FLAGS.train_episodes,
        evaluate_every=FLAGS.evaluate_every,
        eval_episodes=FLAGS.eval_episodes,
        verbose=1,  # Assuming verbose=1 for now, could be a flag
    )


if __name__ == "__main__":
    app.run(main)
