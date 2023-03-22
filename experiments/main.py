from absl import app
from absl import flags
from bsuite.environments import catch, cartpole, mountain_car, bandit
import jax.numpy as jnp
import collections
from typing import NamedTuple
import tensorflow as tf

from rl_lib.utils import NN, build_signature
from rl_lib.experiment import run_loop
from rl_lib.learners import SoftWatkins, QLearner, EMATargetNet
from rl_lib.accumulators import OnlineAccumulator, MultiStepOnlineAccumulator, ReverbAccumulator

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("train_episodes", 1000, "Number of train episodes.")
flags.DEFINE_integer("num_hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_float("kappa", 0.01, "")
flags.DEFINE_float("_lambda", 1.0, "")
flags.DEFINE_float("epsilon", 0.01, "Epsilon-greedy exploration probability.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags.DEFINE_integer("batch_size", 1, "Batch size.")
flags.DEFINE_integer("n_multistep", 1, "n_multistep.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50,
                     "Number of episodes between evaluations.")

def main(unused_arg):
    env = catch.Catch(seed=FLAGS.seed)
    # env = cartpole.Cartpole(seed=FLAGS.seed)
    # env = mountain_car.MountainCar(seed=FLAGS.seed)
    # env = bandit.SimpleBandit()
    network = NN(width=FLAGS.num_hidden_units, n_dense=2, output_dims=env.action_spec().num_values)
    
    agent = SoftWatkins(
        network=network,
        observation_spec=env.observation_spec(),
        learning_rate=FLAGS.learning_rate,
        # ema_decay=0.99,
        kappa=FLAGS.kappa,
        _lambda=FLAGS._lambda,
    )

    # accumulator = OnlineAccumulator()
    # accumulator = MultiStepOnlineAccumulator()

    signature = build_signature(env.observation_spec(), env.action_spec(), FLAGS.n_multistep)
    accumulator = ReverbAccumulator(
        port=5000, 
        signature=signature,
        max_size=1000,
        n_multistep=FLAGS.n_multistep, 
        batch_size=FLAGS.batch_size, 
        shuffle_buffer_size=1000)

    run_loop(
        agent=agent,
        environment=env,
        accumulator=accumulator,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        train_episodes=FLAGS.train_episodes,
        evaluate_every=FLAGS.evaluate_every,
        eval_episodes=FLAGS.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
