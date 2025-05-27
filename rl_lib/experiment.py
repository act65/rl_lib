"""Provides the main experiment loop for running Reinforcement Learning agents.

This module contains the `run_loop` function, which orchestrates the
interaction between an RL agent, an environment, and an experience accumulator
to conduct training and evaluation.
"""

# import haiku as hk # No longer needed for PRNGSequence
import jax
import dm_env

from rl_lib.learners import RLAgent
from rl_lib.accumulators import Accumulator
from typing import Any, Dict # Added Dict for TrainState if more specific types aren't imported

# Assuming these types are defined in learners.py or are general PyTrees/Dicts
# from rl_lib.learners import TrainState, ActorState, Action, Loss
Loss = Any # Placeholder if not imported
TrainState = Dict[str, Any] # More specific if defined by agents
ActorState = Any
ActionType = Any # Placeholder for agent's action output

import gymnasium as gym
import dm_env # Keep for TimeStep structure
from dm_env import specs # For specs.Array
import numpy as np # For type conversion if needed

# Define a simple TimeStep structure if not importing all of dm_env
# This helps in adapting gym's output to what run_loop expects.
class TimeStep(dm_env.TimeStep):
    # We can use dm_env.TimeStep directly if dm_env is a dependency.
    # If trying to minimize dm_env, a custom namedtuple would work:
    # from collections import namedtuple
    # TimeStep = namedtuple('TimeStep', ['step_type', 'reward', 'discount', 'observation'])
    pass


def run_loop(
    agent: RLAgent,
    environment: gym.Env, # Changed to gym.Env
    accumulator: Accumulator,
    seed: int,
    batch_size: int,
    train_episodes: int,
    evaluate_every: int,
    eval_episodes: int,
    verbose: int,
) -> None:
    key = jax.random.PRNGKey(seed)
    key, agent_init_key = jax.random.split(key)
    train_state: TrainState = agent.initial_train_state(agent_init_key)

    print(f"Training agent for {train_episodes} episodes")
    for episode in range(train_episodes):
        obs, info = environment.reset(seed=int(jax.random.randint(key, (), 0, 1000000))) # Seed env for reproducibility
        key, _ = jax.random.split(key) # Consume key

        # Adapt gym's initial observation to a TimeStep
        current_timestep = TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None, # No reward at the first step
            discount=None, # No discount at the first step
            observation=obs
        )

        accumulator.push(
            state=current_timestep.observation, # s_0
            action=None,
            reward=0.0, # Placeholder for initial push
            discount=1.0, # Placeholder for initial push
            next_state=current_timestep.observation # s_0
        )
        actor_state: ActorState = agent.initial_actor_state()
        
        episode_loss_sum = 0.0
        episode_steps = 0

        while not current_timestep.last():
            key, actor_key, learner_key = jax.random.split(key, 3)

            actor_output: ActionType
            actor_output, actor_state = agent.actor_step(
                train_state,
                current_timestep.observation,
                actor_state,
                actor_key,
                evaluation=False,
            )
            
            action_for_env: int
            try:
                # Ensure action is a Python int for gym environment
                action_for_env = int(np.asarray(actor_output))
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not convert actor_output '{actor_output}' to int. Error: {e}")
                raise e

            next_obs, reward, terminated, truncated, info = environment.step(action_for_env)
            done = terminated or truncated

            # Adapt gym's output to a TimeStep
            if done:
                next_timestep_type = dm_env.StepType.LAST
                discount_val = 0.0 # No discount on terminal state
            else:
                next_timestep_type = dm_env.StepType.MID
                discount_val = 1.0 # Assuming discount factor of 1.0 if not terminal, or use a fixed gamma

            next_timestep = TimeStep(
                step_type=next_timestep_type,
                reward=float(reward),
                discount=discount_val,
                observation=next_obs
            )
            
            # Accumulator expects (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1})
            # current_timestep.observation is s_t
            # actor_output is a_t
            # next_timestep.reward is r_{t+1}
            # next_timestep.discount is d_{t+1} (0.0 if terminal, 1.0 or gamma otherwise)
            # next_timestep.observation is s_{t+1}
            accumulator.push(
                state=current_timestep.observation, # s_t
                action=actor_output,               # a_t
                reward=next_timestep.reward,       # r_{t+1}
                discount=next_timestep.discount,   # d_{t+1}
                next_state=next_timestep.observation, # s_{t+1}
            )
            current_timestep = next_timestep

            if accumulator.is_ready(batch_size):
                sampled_batch = accumulator.sample(batch_size)
                train_state, loss_value = agent.learner_step(
                    train_state, sampled_batch, learner_key
                )
                episode_loss_sum += float(loss_value)
                episode_steps += 1
        
        # ... (rest of the loop for verbose printing and evaluation) ...
        if verbose == 1 and episode_steps > 0:
            avg_loss = episode_loss_sum / episode_steps
            print(f"Episode {episode:4d}: Avg Loss: {avg_loss:.3f}")
        elif verbose == 1 and episode_steps == 0 and accumulator.is_ready(batch_size):
             print(f"Episode {episode:4d}: No learning steps taken.")


        accumulator.reset()

        if episode % evaluate_every == 0 or episode == train_episodes - 1 :
            total_eval_returns: float = 0.0
            for eval_ep_num in range(eval_episodes):
                eval_obs, eval_info = environment.reset(seed=int(jax.random.randint(key, (), 1000000, 2000000) + eval_ep_num))
                key, _ = jax.random.split(key)
                
                eval_current_timestep = TimeStep(
                    step_type=dm_env.StepType.FIRST, reward=None, discount=None, observation=eval_obs
                )
                eval_actor_state: ActorState = agent.initial_actor_state()
                episode_return: float = 0.0

                while not eval_current_timestep.last():
                    key, eval_actor_key = jax.random.split(key)
                    eval_actor_output: ActionType
                    eval_actor_output, eval_actor_state = agent.actor_step(
                        train_state,
                        eval_current_timestep.observation,
                        eval_actor_state,
                        eval_actor_key,
                        evaluation=True,
                    )
                    try:
                        eval_action_for_env: int = int(np.asarray(eval_actor_output))
                    except (TypeError, ValueError):
                        print(f"Warning (Eval): Could not convert actor_output '{eval_actor_output}' to int.")
                        break 

                    eval_next_obs, eval_reward, eval_terminated, eval_truncated, _ = environment.step(eval_action_for_env)
                    eval_done = eval_terminated or eval_truncated
                    
                    eval_next_timestep_type = dm_env.StepType.LAST if eval_done else dm_env.StepType.MID
                    eval_discount_val = 0.0 if eval_done else 1.0

                    eval_current_timestep = TimeStep(
                        step_type=eval_next_timestep_type,
                        reward=float(eval_reward),
                        discount=eval_discount_val,
                        observation=eval_next_obs
                    )
                    if eval_current_timestep.reward is not None: # Should always be float now
                        episode_return += eval_current_timestep.reward
                total_eval_returns += episode_return
            
            avg_returns: float = total_eval_returns / eval_episodes if eval_episodes > 0 else 0.0
            print(f"Episode {episode:4d}: Average Evaluation Returns: {avg_returns:.2f}")