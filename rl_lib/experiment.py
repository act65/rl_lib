"""Provides the main experiment loop for running Reinforcement Learning agents.

This module contains the `run_loop` function, which orchestrates the
interaction between an RL agent, an environment, and an experience accumulator
to conduct training and evaluation.
"""

import haiku as hk
import jax
import dm_env  # For dm_env.Environment and dm_env.TimeStep

from rl_lib.learners import RLAgent  # Assuming RLAgent is in rl_lib.learners
from rl_lib.accumulators import (
    Accumulator,
)  # Assuming Accumulator is in rl_lib.accumulators
from typing import Any  # For TrainState, ActorState, ActorOutput etc.

# If TrainState, ActorState, Action, Loss are defined types in learners, import them too.
# For now, using Any or more generic types.
# from rl_lib.learners import TrainState, ActorState, Action, Loss


def run_loop(
    agent: RLAgent,
    environment: dm_env.Environment,  # Or Any if dm_env is not strictly used
    accumulator: Accumulator,
    seed: int,
    batch_size: int,
    train_episodes: int,
    evaluate_every: int,
    eval_episodes: int,
    verbose: int,  # Assuming int, could be bool if only 0 or 1
) -> None:
    """Orchestrates the training and evaluation of a Reinforcement Learning agent.

    This function manages the main loop of an RL experiment. It handles:
    - Initialization of the agent and random number generators.
    - Running a specified number of training episodes.
    - Within each training episode:
        - Resetting the environment and accumulator.
        - Stepping the agent and environment (acting and interaction).
        - Accumulating experiences.
        - Triggering the agent's learning step when the accumulator is ready.
    - Periodically evaluating the agent's performance.
    - Printing training and evaluation metrics based on verbosity.

    Args:
        agent: An RL agent instance conforming to the `RLAgent` interface (or similar),
            with methods like `initial_train_state`, `initial_actor_state`,
            `actor_step`, and `learner_step`.
        environment: An environment instance, typically conforming to a standard
            RL environment interface like dm_env, with `reset()` and `step()` methods.
        accumulator: An experience replay buffer or accumulator instance,
            with `push()`, `sample()`, `is_ready()`, and `reset()` methods.
        seed (int): An integer seed for initializing JAX's pseudo-random
            number generator (PRNG).
        batch_size (int): The batch size to be used for sampling from the
            accumulator and for the agent's learning step.
        train_episodes (int): The total number of episodes to run for training.
        evaluate_every (int): The frequency (in episodes) at which to evaluate
            the agent. Evaluation is run every `evaluate_every` episodes.
        eval_episodes (int): The number of episodes to run for each evaluation phase.
        verbose (int): Controls the verbosity of logging. If 1, prints loss
            during training. Evaluation returns are always printed.
                       Other values might be supported for less/more detail.

    Returns:
        None. This function currently does not return any value. Results are
        printed to standard output.
    """

    # Init agent and PRNG sequence.
    rng: hk.PRNGSequence = hk.PRNGSequence(jax.random.PRNGKey(seed))
    # Assuming TrainState is a Dict or PyTree, Any is a safe bet if not specifically defined
    train_state: Any = agent.initial_train_state(next(rng))

    print(f"Training agent for {train_episodes} episodes")
    for episode in range(train_episodes):

        # Prepare agent, environment and accumulator for a new episode.
        timestep: dm_env.TimeStep = environment.reset()
        # For the first push, action and reward are typically None or placeholders
        accumulator.push(timestep.observation, action=None, reward=timestep.reward, discount=timestep.discount, next_state=timestep.observation)  # type: ignore
        actor_state: Any = agent.initial_actor_state()  # ActorState type if defined

        while not timestep.last():

            # Acting.
            # Assuming actor_output is an Action type, actor_state is ActorState
            actor_output: Any
            actor_output, actor_state = agent.actor_step(
                train_state,
                timestep.observation,
                actor_state,
                next(rng),
                evaluation=False,
            )

            # Agent-environment interaction.
            action: int = int(actor_output)  # Assuming actor_output can be cast to int
            timestep = environment.step(action)

            # Accumulate experience.
            # next_state is timestep.observation from the latest step
            accumulator.push(
                timestep.observation,
                action,
                timestep.reward,
                timestep.discount,
                next_state=timestep.observation,
            )

            # Learning.
            if accumulator.is_ready(batch_size):
                loss: float  # Or jax.Array if it's a JAX scalar
                train_state, loss = agent.learner_step(
                    train_state, accumulator.sample(batch_size), next(rng)
                )

                if verbose == 1:
                    # Ensure loss is a scalar float for printing, or convert if it's a JAX array
                    print(f"Episode {episode:4d}: Loss: {float(loss):.2f}")

        accumulator.reset()

        # Evaluation.
        if not episode % evaluate_every:
            returns: float = 0.0
            for _ in range(eval_episodes):
                eval_timestep: dm_env.TimeStep = environment.reset()
                eval_actor_state: Any = agent.initial_actor_state()

                while not eval_timestep.last():
                    eval_actor_output: Any
                    eval_actor_output, eval_actor_state = agent.actor_step(
                        train_state,
                        eval_timestep.observation,
                        eval_actor_state,
                        next(rng),
                        evaluation=True,
                    )
                    eval_action: int = int(eval_actor_output)
                    eval_timestep = environment.step(eval_action)
                    if eval_timestep.reward is not None:
                        returns += eval_timestep.reward

            avg_returns: float = returns / eval_episodes
            print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")
