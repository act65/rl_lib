"""Experiment loop."""

import haiku as hk
import jax

def run_loop(
        agent, environment, accumulator, seed,
        batch_size, train_episodes, evaluate_every, eval_episodes, verbose):
    """A simple run loop for examples of reinforcement learning with rlax."""

    # Init agent.
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    train_state = agent.initial_train_state(next(rng))

    print(f"Training agent for {train_episodes} episodes")
    for episode in range(train_episodes):

        # Prepare agent, environment and accumulator for a new episode.
        timestep = environment.reset()
        accumulator.push(timestep.observation, None, timestep.reward, timestep.discount)
        actor_state = agent.initial_actor_state()

        while not timestep.last():

            # Acting.
            actor_output, actor_state = agent.actor_step(
                train_state, timestep.observation, actor_state, next(rng), evaluation=False)

            # Agent-environment interaction.
            action = int(actor_output)
            timestep = environment.step(action)

            # Accumulate experience.
            accumulator.push(timestep.observation, action, timestep.reward, timestep.discount)

            # Learning.
            if accumulator.is_ready(batch_size):
                train_state, loss = agent.learner_step(
                train_state, accumulator.sample(batch_size), next(rng))

                if verbose == 1:
                    print(f"Episode {episode:4d}: Loss: {loss:.2f}")

        accumulator.reset()

        # Evaluation.
        if not episode % evaluate_every:
            returns = 0.
            for _ in range(eval_episodes):
                timestep = environment.reset()
                actor_state = agent.initial_actor_state()

                while not timestep.last():
                    actor_output, actor_state = agent.actor_step(
                        train_state, timestep.observation, actor_state, next(rng), evaluation=True)
                    timestep = environment.step(int(actor_output))
                    returns += timestep.reward

            avg_returns = returns / eval_episodes
            print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")