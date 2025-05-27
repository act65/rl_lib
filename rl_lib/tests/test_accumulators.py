from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
import numpy as np
import gymnasium as gym # Changed from bsuite
import reverb # Ensure reverb is imported for server stop

from rl_lib.utils import build_signature
from rl_lib.accumulators import (
    OnlineAccumulator,
    MultiStepOnlineAccumulator,
    ReverbAccumulator,
    MultiAgentReverbAccumulator,
    RLPDAccumulator,
    MultiAgentRLPDAccumulator,
)
from rl_lib.experiment import run_loop
from rl_lib.learners import Random


# Helper Spec classes for Gymnasium (same as in test_learners)
class GymObservationSpec:
    def __init__(self, observation_space: gym.spaces.Space):
        self.shape = observation_space.shape
        self.dtype = observation_space.dtype if hasattr(observation_space, 'dtype') else jnp.float32
        if not self.shape:
             if hasattr(observation_space, 'n'):
                 self.shape = (1,)
                 self.dtype = jnp.int32

    def generate_value(self):
        return jnp.zeros(self.shape, dtype=self.dtype)

class GymActionSpec:
    def __init__(self, action_space: gym.spaces.Discrete):
        self.shape = ()
        self.dtype = jnp.int32
        self.num_values = int(action_space.n)

    def generate_value(self):
        return jnp.array(0, dtype=self.dtype)

class TestSimpleAccumulators(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.env_for_spec = gym.make("CartPole-v1")
        obs_space = self.env_for_spec.observation_space
        action_space = self.env_for_spec.action_space
        self.obs_spec = GymObservationSpec(obs_space)
        self.action_spec = GymActionSpec(action_space)
        self.env_for_spec.close()

        # Define a clear (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}) transition
        self.s_t = jnp.array(obs_space.sample(), dtype=self.obs_spec.dtype) * 0.5 # Ensure some values
        self.a_t = jnp.array(action_space.sample(), dtype=self.action_spec.dtype)
        self.r_tp1 = 0.5
        self.d_tp1 = 0.99
        self.s_tp1 = jnp.array(obs_space.sample(), dtype=self.obs_spec.dtype) * 0.8 # Ensure different from s_t

    def test_online_accumulator(self):
        acc = OnlineAccumulator(max_size=1)
        self.assertFalse(acc.is_ready(1))

        # Push an incomplete transition (action is None)
        acc.push(self.s_t, None, 0.0, 1.0, next_state=self.s_tp1)
        self.assertFalse(acc.is_ready(1)) # Buffer not set

        # Push a complete transition (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1})
        acc.push(
            self.s_t, self.a_t, self.r_tp1, self.d_tp1, next_state=self.s_tp1,
        )
        self.assertTrue(acc.is_ready(1))
        sample = acc.sample(1) # Sample is a dict with batched (B=1) JAX arrays

        np.testing.assert_array_equal(sample["states"][0], self.s_t)
        np.testing.assert_array_equal(sample["actions"][0], self.a_t)
        np.testing.assert_allclose(sample["rewards"][0], self.r_tp1, rtol=1e-6)
        np.testing.assert_allclose(sample["discounts"][0], self.d_tp1, rtol=1e-6)
        np.testing.assert_array_equal(sample["next_states"][0], self.s_tp1)

        acc.reset()
        self.assertFalse(acc.is_ready(1))

    def test_multistep_online_accumulator(self):
        n_multistep = 3
        acc = MultiStepOnlineAccumulator(n_multistep=n_multistep)
        self.assertFalse(acc.is_ready(1)) # Buffer not full yet

        pushed_transitions_for_assertion = []
        
        # s_curr is s_t for the current loop iteration
        s_curr = jnp.array(self.env_for_spec.observation_space.sample(), dtype=self.obs_spec.dtype) * 0.1

        for i in range(n_multistep):
            # Define the transition (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}) for this step
            s_t_loop = s_curr
            a_t_loop = jnp.array(self.env_for_spec.action_space.sample() + i, dtype=self.action_spec.dtype)
            r_tp1_loop = self.r_tp1 + float(i) * 0.1
            d_tp1_loop = self.d_tp1
            s_tp1_loop = jnp.array(self.env_for_spec.observation_space.sample(), dtype=self.obs_spec.dtype) * (0.2 + i*0.1)

            acc.push(s_t_loop, a_t_loop, r_tp1_loop, d_tp1_loop, next_state=s_tp1_loop)
            pushed_transitions_for_assertion.append((s_t_loop, a_t_loop, r_tp1_loop, d_tp1_loop, s_tp1_loop))
            
            s_curr = s_tp1_loop # s_{t+1} becomes s_t for the next iteration

            if i < n_multistep - 1:
                self.assertFalse(acc.is_ready(1), f"Should not be ready at step {i}")
        
        self.assertTrue(acc.is_ready(1), f"Buffer size: {len(acc.buffer)}, expected: {n_multistep}")
        
        # The sample will contain all `n_multistep` transitions that are currently in the buffer
        sample = acc.sample(1) 
        
        self.assertCountEqual(sample.keys(), acc.names)
        for key in sample:
            self.assertEqual(sample[key].shape, (1, n_multistep) + pushed_transitions_for_assertion[0][0].shape if key in ["states", "next_states"] else (1,n_multistep) + pushed_transitions_for_assertion[0][1].shape, f"Shape mismatch for key {key}")


        # Construct expected sample from the pushed_transitions_for_assertion
        # The sample should match exactly what was in the buffer when is_ready became true.
        expected_states_in_sample = jnp.stack([item[0] for item in pushed_transitions_for_assertion])
        expected_actions_in_sample = jnp.stack([item[1] for item in pushed_transitions_for_assertion])
        expected_rewards_in_sample = jnp.stack([item[2] for item in pushed_transitions_for_assertion])
        expected_discounts_in_sample = jnp.stack([item[3] for item in pushed_transitions_for_assertion])
        expected_next_states_in_sample = jnp.stack([item[4] for item in pushed_transitions_for_assertion])

        np.testing.assert_array_equal(sample["states"][0], expected_states_in_sample)
        np.testing.assert_array_equal(sample["actions"][0], expected_actions_in_sample)
        np.testing.assert_allclose(sample["rewards"][0], expected_rewards_in_sample, rtol=1e-6)
        np.testing.assert_allclose(sample["discounts"][0], expected_discounts_in_sample, rtol=1e-6)
        np.testing.assert_array_equal(sample["next_states"][0], expected_next_states_in_sample)
        
        # After sample(), one item is popped from the left of the deque
        self.assertEqual(len(acc.buffer), n_multistep - 1, "Buffer size after sample is not as expected.")
        acc.reset()
        self.assertFalse(acc.is_ready(1))

class TestReverb(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.env = gym.make("CartPole-v1") # Use Gymnasium env
        self.obs_spec = GymObservationSpec(self.env.observation_space)
        self.action_spec = GymActionSpec(self.env.action_space)
        self.signature_n1 = build_signature(self.obs_spec, self.action_spec, n_multistep=1)
        # Ensure env is closed if only used for specs
        self.env.close()


    @parameterized.parameters(
        {"batch_size": 1, "n_multistep": 1, "port": 5000},
        {"batch_size": 4, "n_multistep": 1, "port": 5001},
        {"batch_size": 1, "n_multistep": 4, "port": 5002},
        {"batch_size": 4, "n_multistep": 4, "port": 5003},
    )
    def test_reverb_sample(self, batch_size, n_multistep, port):
        env_for_run = gym.make("CartPole-v1") # Fresh env for run_loop
        obs_spec_run = GymObservationSpec(env_for_run.observation_space)
        action_spec_run = GymActionSpec(env_for_run.action_space)

        signature = build_signature(obs_spec_run, action_spec_run, n_multistep)
        accumulator = ReverbAccumulator(
            port=port, signature=signature, max_size=100, n_multistep=n_multistep,
            batch_size=batch_size, shuffle_buffer_size=50, start_server_flag=True,
        )
        try:
            # CartPole episodes can be up to 500 steps.
            # Need enough steps to form `batch_size` trajectories of `n_multistep`.
            # Total steps needed approx: (batch_size + n_multistep -1) * n_multistep for worst case alignment
            # Or more simply, ensure enough total transitions are generated.
            # If an episode is ~20 steps on average for random agent.
            # train_episodes = (batch_size * n_multistep // 20) + n_multistep + 5
            train_episodes = 10 # Increased episodes to ensure buffer fills

            run_loop(
                agent=Random(action_spec=action_spec_run), environment=env_for_run,
                accumulator=accumulator, seed=port, batch_size=batch_size, # Seed with port for variation
                train_episodes=train_episodes, evaluate_every=train_episodes + 1,
                eval_episodes=0, verbose=0,
            )

            self.assertTrue(accumulator.is_ready(batch_size),
                f"Accumulator not ready. Size: {accumulator.current_size}, Batch: {batch_size}, Port: {port}")
            batch = accumulator.sample(batch_size)
            for k, v in batch.items():
                self.assertEqual(v.shape[:2], (batch_size, n_multistep))
        finally:
            if accumulator.server:
                try:
                    accumulator.server.stop()
                except Exception as e:
                    print(f"Error stopping server on port {port}: {e}")
            env_for_run.close()


    @parameterized.parameters(
        {"n_agents": 2, "n_multistep": 4, "batch_size_per_agent": 1, "port": 5005},
        {"n_agents": 3, "n_multistep": 2, "batch_size_per_agent": 2, "port": 5006},
    )
    def test_multiagent(self, n_agents, n_multistep, batch_size_per_agent, port):
        # For MA tests, use specs from a single env instance
        env_spec_source = gym.make("CartPole-v1")
        obs_spec_ma = GymObservationSpec(env_spec_source.observation_space)
        action_spec_ma = GymActionSpec(env_spec_source.action_space)
        env_spec_source.close()

        signature = build_signature(obs_spec_ma, action_spec_ma, n_multistep)
        # Total batch size for sampling from the single table
        total_batch_size = batch_size_per_agent * n_agents

        accumulator = MultiAgentReverbAccumulator(
            port=port, signature=signature, max_size=n_agents * (n_multistep + 10),
            n_multistep=n_multistep, batch_size=total_batch_size,
            shuffle_buffer_size=n_agents * (n_multistep + 10), start_server_flag=True,
        )
        try:
            obs = {str(i): obs_spec_ma.generate_value() for i in range(n_agents)}
            acts = {str(i): action_spec_ma.generate_value() for i in range(n_agents)}
            rews = {str(i): 0.0 for i in range(n_agents)}
            discounts = {str(i): 1.0 for i in range(n_agents)}

            # Push enough steps for each agent's writer to form at least batch_size_per_agent items
            # Each writer needs to create `batch_size_per_agent` items.
            # Each item requires `n_multistep` steps.
            # Total pushes needed: `batch_size_per_agent + n_multistep -1` for each agent to have enough history.
            # Let's simplify: push `batch_size_per_agent + n_multistep` times.
            for _ in range(batch_size_per_agent + n_multistep + 2): # Add some buffer
                accumulator.push(obs, acts, rews, discounts, next_state=obs)

            # Each agent's writer contributes items to the same table.
            # Expected items = n_agents * batch_size_per_agent (if each writer creates that many)
            # This check is tricky because writers might create items at different rates.
            # A simpler check is if the table has enough for the total_batch_size.
            self.assertTrue(accumulator.current_size >= total_batch_size,
                            f"MA Reverb not filled enough. Has: {accumulator.current_size}, needs: {total_batch_size}")

            # Sample and check batch shape
            if accumulator.current_size >= total_batch_size:
                sample = accumulator.sample(total_batch_size)
                for k_sample, v_sample in sample.items():
                    self.assertEqual(v_sample.shape[:2], (total_batch_size, n_multistep))

        finally:
            if accumulator.server:
                try:
                    accumulator.server.stop()
                except Exception as e:
                    print(f"Error stopping server on port {port}: {e}")


    def test_rlpd_accumulator_push_sample(self):
        n_multistep = 1
        batch_size = 2 # 1 online, 1 offline
        port = 5007

        env_for_run = gym.make("CartPole-v1")
        obs_spec_rlpd = GymObservationSpec(env_for_run.observation_space)
        action_spec_rlpd = GymActionSpec(env_for_run.action_space)
        signature_rlpd = build_signature(obs_spec_rlpd, action_spec_rlpd, n_multistep)


        accumulator = RLPDAccumulator(
            port=port, signature=signature_rlpd, max_size=100, n_multistep=n_multistep,
            batch_size=batch_size, shuffle_buffer_size=50, start_server_flag=True,
        )
        try:
            # Push to online and offline tables
            for table_name in ["online", "offline"]:
                obs = env_for_run.reset()[0] # Gym returns obs, info
                accumulator.push(obs, None, 0.0, 0.0, table_name=table_name, next_state=obs)
                for _ in range(batch_size // 2 * n_multistep + 10): # Ensure enough items
                    action = env_for_run.action_space.sample()
                    next_obs, reward, terminated, truncated, _ = env_for_run.step(action)
                    done = terminated or truncated
                    discount = 0.99 * (1-float(done))
                    accumulator.push(obs, action, float(reward), discount, table_name=table_name, next_state=next_obs)
                    obs = next_obs
                    if done:
                        obs = env_for_run.reset()[0]

            self.assertTrue(accumulator.is_ready(batch_size), f"RLPD Accumulator not ready. Status: {accumulator.status}")
            sample = accumulator.sample(batch_size)
            for k, v in sample.items():
                self.assertEqual(v.shape[:2], (batch_size, n_multistep))
        finally:
            if accumulator.server:
                try:
                    accumulator.server.stop()
                except Exception as e:
                    print(f"Error stopping server on port {port}: {e}")
            env_for_run.close()


    def test_multi_agent_rlpd_accumulator(self):
        n_agents = 2
        n_multistep = 1
        batch_size = n_agents * 2 # 2 per agent, 1 online, 1 offline per agent effectively
        port = 5008

        env_spec_source = gym.make("CartPole-v1")
        obs_spec_marlpd = GymObservationSpec(env_spec_source.observation_space)
        action_spec_marlpd = GymActionSpec(env_spec_source.action_space)
        env_spec_source.close()
        signature_marlpd = build_signature(obs_spec_marlpd, action_spec_marlpd, n_multistep)

        accumulator = MultiAgentRLPDAccumulator(
            port=port, signature=signature_marlpd, max_size=100 * n_agents, n_multistep=n_multistep,
            batch_size=batch_size, shuffle_buffer_size=50 * n_agents, start_server_flag=True,
        )
        try:
            for table_name in ["online", "offline"]:
                current_obs_multi = {str(i): obs_spec_marlpd.generate_value() for i in range(n_agents)}
                dummy_actions_none = {str(i): None for i in range(n_agents)}
                dummy_rewards_zero = {str(i): 0.0 for i in range(n_agents)}
                dummy_discounts_zero = {str(i): 0.0 for i in range(n_agents)}
                accumulator.push(
                    current_obs_multi, dummy_actions_none, dummy_rewards_zero,
                    dummy_discounts_zero, table_name=table_name, next_state=current_obs_multi,
                )
                # Steps per agent per table: (batch_size / (2 * n_agents)) + n_multistep + buffer
                steps_per_agent = (batch_size // (2 * n_agents)) * n_multistep + n_multistep + 5
                for _ in range(steps_per_agent):
                    multi_actions = {str(i): action_spec_marlpd.generate_value() for i in range(n_agents)}
                    multi_rewards = {str(i): np.random.rand() for i in range(n_agents)}
                    multi_discounts = {str(i): 0.99 for i in range(n_agents)}
                    next_multi_obs = {str(i): obs_spec_marlpd.generate_value() for i in range(n_agents)}
                    accumulator.push(
                        current_obs_multi, multi_actions, multi_rewards, multi_discounts,
                        table_name=table_name, next_state=next_multi_obs,
                    )
                    current_obs_multi = next_multi_obs

            self.assertTrue(accumulator.is_ready(batch_size), f"MA RLPD Accumulator not ready. Status: {accumulator.status}")
            sample = accumulator.sample(batch_size)
            for k, v in sample.items():
                self.assertEqual(v.shape[:2], (batch_size, n_multistep))
        finally:
            if accumulator.server:
                try:
                    accumulator.server.stop()
                except Exception as e:
                    print(f"Error stopping server on port {port}: {e}")


if __name__ == "__main__":
    absltest.main()