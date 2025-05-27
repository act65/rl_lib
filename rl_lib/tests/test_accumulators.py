from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
import numpy as np
# import collections  # F401: For MultiStepOnlineAccumulator internal buffer type check - Deque is from typing

from bsuite.environments import catch

from rl_lib.utils import build_signature
from rl_lib.accumulators import (
    OnlineAccumulator,
    MultiStepOnlineAccumulator,
    ReverbAccumulator,
    MultiAgentReverbAccumulator,
    RLPDAccumulator,
    MultiAgentRLPDAccumulator,
)
from rl_lib.experiment import run_loop  # Used in existing Reverb test
from rl_lib.learners import Random  # Used in existing Reverb test


class TestSimpleAccumulators(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.obs_spec_shape = (4,)
        self.action_spec_num_values = 2
        self.dummy_obs = jnp.zeros(self.obs_spec_shape, dtype=jnp.float32)
        self.dummy_next_obs = jnp.ones(self.obs_spec_shape, dtype=jnp.float32)
        self.dummy_action = 1
        self.dummy_reward = 0.5
        self.dummy_discount = 0.99

    def test_online_accumulator(self):
        acc = OnlineAccumulator(max_size=1)
        self.assertFalse(acc.is_ready(1))

        # Initial push, action is None
        acc.push(
            self.dummy_obs, None, 0.0, 0.0, next_state=self.dummy_obs
        )  # Pass dummy_obs as next_state for consistency
        self.assertFalse(
            acc.is_ready(1),
            "Accumulator should not be ready after initial push with None action.",
        )

        # Valid push
        acc.push(
            self.dummy_next_obs,
            self.dummy_action,
            self.dummy_reward,
            self.dummy_discount,
            next_state=self.dummy_next_obs,
        )
        self.assertTrue(acc.is_ready(1))

        sample = acc.sample(1)
        self.assertCountEqual(
            sample.keys(), ["states", "actions", "rewards", "discounts", "next_states"]
        )
        np.testing.assert_array_equal(sample["states"], self.dummy_obs)
        np.testing.assert_array_equal(sample["actions"], self.dummy_action)
        np.testing.assert_array_equal(sample["rewards"], self.dummy_reward)
        np.testing.assert_array_equal(sample["discounts"], self.dummy_discount)
        np.testing.assert_array_equal(sample["next_states"], self.dummy_next_obs)

        acc.reset()
        self.assertFalse(acc.is_ready(1))
        self.assertIsNone(acc.buffer)
        self.assertIsNone(acc.prev_state)

    def test_multistep_online_accumulator(self):
        n_multistep = 3
        acc = MultiStepOnlineAccumulator(n_multistep=n_multistep)
        self.assertFalse(acc.is_ready(1))

        # Push initial state
        initial_obs_for_sequence = jnp.full(
            self.obs_spec_shape, -1.0
        )  # Distinct initial obs
        acc.push(
            initial_obs_for_sequence,
            None,
            0.0,
            0.0,
            next_state=initial_obs_for_sequence,
        )

        pushed_items = []
        # First item in buffer will use initial_obs_for_sequence as prev_state
        # prev_obs_for_push = initial_obs_for_sequence # F841: local variable 'prev_obs_for_push' is assigned to but never used

        for i in range(n_multistep):
            obs_i = jnp.full(
                self.obs_spec_shape, float(i)
            )  # Current obs becomes next_state for the pushed item
            action_i = self.dummy_action + i
            reward_i = self.dummy_reward + i

            acc.push(obs_i, action_i, reward_i, self.dummy_discount, next_state=obs_i)

            # Store what should be in the buffer
            # (prev_state, action, reward, discount, current_state_which_is_next_state_for_transition)
            if i == 0:  # First valid push uses initial_obs_for_sequence as prev_state
                pushed_items.append(
                    (
                        initial_obs_for_sequence,
                        action_i,
                        reward_i,
                        self.dummy_discount,
                        obs_i,
                    )
                )
            else:  # Subsequent pushes use the obs_i from the previous iteration
                pushed_items.append(
                    (
                        jnp.full(self.obs_spec_shape, float(i - 1)),
                        action_i,
                        reward_i,
                        self.dummy_discount,
                        obs_i,
                    )
                )

            if i < n_multistep - 1:
                self.assertFalse(acc.is_ready(1))

        self.assertTrue(acc.is_ready(1))
        self.assertEqual(len(acc.buffer), n_multistep)

        sample = acc.sample(1)
        self.assertCountEqual(
            sample.keys(), ["states", "actions", "rewards", "discounts", "next_states"]
        )

        for key in sample:
            self.assertEqual(sample[key].shape[0], 1)  # Batch size
            self.assertEqual(sample[key].shape[1], n_multistep)  # Sequence length

        # Check content of the sample (first item pushed is the first in sequence)
        expected_states_in_sample = jnp.stack([item[0] for item in pushed_items])  # s_t
        expected_actions_in_sample = jnp.stack(
            [item[1] for item in pushed_items]
        )  # a_t
        expected_next_states_in_sample = jnp.stack(
            [item[4] for item in pushed_items]
        )  # s_{t+1}

        np.testing.assert_array_equal(sample["states"][0], expected_states_in_sample)
        np.testing.assert_array_equal(sample["actions"][0], expected_actions_in_sample)
        np.testing.assert_array_equal(
            sample["next_states"][0], expected_next_states_in_sample
        )

        self.assertEqual(
            len(acc.buffer),
            n_multistep - 1,
            "Buffer size should decrease by 1 after sampling.",
        )

        acc.reset()
        self.assertEqual(len(acc.buffer), 0)
        self.assertIsNone(acc.prev_state)


class TestReverb(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.env = catch.Catch(seed=0)
        self.obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()
        # Signature for single step transitions, will be used for RLPD tests
        self.signature_n1 = build_signature(
            self.obs_spec, self.action_spec, n_multistep=1
        )

    @parameterized.parameters(
        {"batch_size": 1, "n_multistep": 1, "port": 5000},
        {"batch_size": 4, "n_multistep": 1, "port": 5001},
        {"batch_size": 1, "n_multistep": 4, "port": 5002},
        {"batch_size": 4, "n_multistep": 4, "port": 5003},
    )
    def test_reverb_sample(self, batch_size, n_multistep, port):
        # Use a unique port for each parameterized test to avoid conflicts
        signature = build_signature(
            self.env.observation_spec(), self.env.action_spec(), n_multistep
        )
        accumulator = ReverbAccumulator(
            port=port,
            signature=signature,
            max_size=20,  # Increased max_size to accommodate more diverse test cases
            n_multistep=n_multistep,
            batch_size=batch_size,
            shuffle_buffer_size=20,  # Increased shuffle buffer
            start_server_flag=True,
        )  # Ensure server is started for test

        # Run enough episodes to ensure the buffer can be filled sufficiently
        # Total items to generate: batch_size * n_multistep + some_buffer_for_multi_step_logic
        # Each episode is 10 steps in Catch env.
        # Min items for 1 sample: n_multistep steps for trajectory + (n_multistep-1) for history
        # For ReplayWriter to create an item: episode_steps >= n_multistep
        # To ensure `is_ready(batch_size)` is true, table size must be >= batch_size
        # Each call to run_loop with 10 episodes will generate 10*10 = 100 transitions
        # Each trajectory is n_multistep.
        # Need at least batch_size trajectories.
        # Number of transitions needed to form `batch_size` trajectories of `n_multistep`
        # is roughly `batch_size * n_multistep`.
        # The run_loop pushes one initial item, then `action, reward, discount, next_obs`
        # This means for each env step, one item is pushed to ReplayWriter's history.
        # ReplayWriter creates an item when its history has >= n_multistep items.

        # Let's simplify: ensure enough items are pushed to form `batch_size` distinct trajectories.
        # Each trajectory requires `n_multistep` steps to be observed by the writer.
        # To be safe, run for enough steps to generate `batch_size + n_multistep` items.
        # Catch episodes are short (10 steps).
        required_items = batch_size + n_multistep
        train_episodes = (required_items // 10) + 2  # Ensure enough steps

        run_loop(
            agent=Random(self.env.action_spec()),
            environment=self.env,
            accumulator=accumulator,
            seed=0,  # Use a fixed seed for reproducibility
            batch_size=batch_size,
            train_episodes=train_episodes,
            evaluate_every=train_episodes + 1,  # No evaluation during this test
            eval_episodes=0,
            verbose=0,
        )

        self.assertTrue(
            accumulator.is_ready(batch_size),
            f"Accumulator not ready. Size: {accumulator.current_size}, Batch: {batch_size}",
        )
        batch = accumulator.sample(batch_size)
        for k, v in batch.items():
            self.assertEqual(v.shape[0], batch_size)
            self.assertEqual(v.shape[1], n_multistep)

        if accumulator.server:  # Stop server if this test instance started it
            accumulator.server.stop()

    @parameterized.parameters(
        {"n_agents": 2, "n_multistep": 4, "batch_size": 1, "port": 5005},
        {"n_agents": 5, "n_multistep": 4, "batch_size": 1, "port": 5006},
    )
    def test_multiagent(self, n_agents, n_multistep, batch_size, port):
        signature = build_signature(
            self.env.observation_spec(), self.env.action_spec(), n_multistep
        )
        accumulator = MultiAgentReverbAccumulator(
            port=port,
            signature=signature,
            max_size=n_agents * (n_multistep + 5),  # Ensure enough space
            n_multistep=n_multistep,
            batch_size=batch_size
            * n_agents,  # Sample a batch per agent for test simplicity
            shuffle_buffer_size=n_agents * (n_multistep + 5),
            start_server_flag=True,
        )

        obs = {
            str(i): self.env.observation_spec().generate_value()
            for i in range(n_agents)
        }
        acts = {
            str(i): self.env.action_spec().generate_value() for i in range(n_agents)
        }
        rews = {str(i): 0.0 for i in range(n_agents)}
        discounts = {str(i): 1.0 for i in range(n_agents)}

        # Each push for MultiAgentReverbAccumulator results in N ReplayWriters pushing.
        # Each ReplayWriter creates an item when its history has >= n_multistep items.
        # We need to push enough steps for each agent's writer to form at least one item.
        for _ in range(
            n_multistep + 1
        ):  # n_multistep for history, +1 to trigger create_item
            accumulator.push(
                obs, acts, rews, discounts, next_state=obs
            )  # Pass next_state explicitly

        # current_size for MA is sum of sizes of tables (only one table here)
        # Each agent's writer contributes items to the same table.
        # After n_multistep+1 pushes, each of the n_agents writers should have created 1 item.
        self.assertEqual(accumulator.current_size, n_agents)

        if accumulator.server:
            accumulator.server.stop()

    def test_rlpd_accumulator_push_sample(self):
        n_multistep = 1
        batch_size = 2  # 1 online, 1 offline
        port = 5007

        accumulator = RLPDAccumulator(
            port=port,
            signature=self.signature_n1,  # n_multistep=1
            max_size=40,
            n_multistep=n_multistep,
            batch_size=batch_size,
            shuffle_buffer_size=20,
            start_server_flag=True,
        )

        # Push to online table
        obs_online = self.env.reset().observation
        accumulator.push(
            obs_online, None, 0.0, 0.0, table_name="online", next_state=obs_online
        )
        for _ in range(
            batch_size // 2 * n_multistep + 5
        ):  # Ensure enough for a few samples
            action = self.env.action_spec().generate_value()
            ts = self.env.step(action)
            accumulator.push(
                ts.observation,
                action,
                ts.reward,
                ts.discount,
                table_name="online",
                next_state=ts.observation,
            )
            if ts.last():
                obs_online = self.env.reset().observation

        # Push to offline table
        obs_offline = self.env.reset().observation
        accumulator.push(
            obs_offline, None, 0.0, 0.0, table_name="offline", next_state=obs_offline
        )
        for _ in range(batch_size // 2 * n_multistep + 5):
            action = self.env.action_spec().generate_value()
            ts = self.env.step(action)
            accumulator.push(
                ts.observation,
                action,
                ts.reward,
                ts.discount,
                table_name="offline",
                next_state=ts.observation,
            )
            if ts.last():
                obs_offline = self.env.reset().observation

        self.assertTrue(
            accumulator.is_ready(batch_size),
            f"RLPD Accumulator not ready. Status (online, offline): {accumulator.status}",
        )

        sample = accumulator.sample(batch_size)
        self.assertCountEqual(sample.keys(), self.signature_n1.keys())
        for k, v in sample.items():
            self.assertEqual(v.shape[0], batch_size)  # Total batch size
            self.assertEqual(v.shape[1], n_multistep)

        online_size, offline_size = accumulator.status
        self.assertTrue(online_size >= batch_size // 2)
        self.assertTrue(offline_size >= batch_size // 2)

        if accumulator.server:
            accumulator.server.stop()

    def test_multi_agent_rlpd_accumulator(self):
        n_agents = 2
        n_multistep = 1
        # Each sample call for RLPD with batch_size X will try to get X/2 from online and X/2 from offline.
        # For multi-agent, these X/2 items can be from any of the agents.
        batch_size = n_agents * 2  # e.g., 2 agents, batch 4 -> 2 online, 2 offline
        port = 5008

        accumulator = MultiAgentRLPDAccumulator(
            port=port,
            signature=self.signature_n1,  # n_multistep=1
            max_size=40 * n_agents,
            n_multistep=n_multistep,
            batch_size=batch_size,
            shuffle_buffer_size=20 * n_agents,
            start_server_flag=True,
        )

        # Generate enough data for each agent for both online and offline tables
        for table_name in ["online", "offline"]:
            # For each agent, generate initial observation
            current_obs_multi = {
                str(i): self.env.reset().observation for i in range(n_agents)
            }
            # Initial push for each agent
            dummy_actions_none = {str(i): None for i in range(n_agents)}
            dummy_rewards_zero = {str(i): 0.0 for i in range(n_agents)}
            dummy_discounts_zero = {str(i): 0.0 for i in range(n_agents)}
            accumulator.push(
                current_obs_multi,
                dummy_actions_none,
                dummy_rewards_zero,
                dummy_discounts_zero,
                table_name=table_name,
                next_state=current_obs_multi,
            )

            # Each agent needs to contribute enough items for batch_size/2 items to be sampled
            # Total items needed per table = batch_size / 2
            # Steps per agent per table: (batch_size / 2 / n_agents) + n_multistep + some buffer
            steps_per_agent = (
                (batch_size // (2 * n_agents)) * n_multistep + n_multistep + 5
            )

            for step_num in range(steps_per_agent):
                multi_actions = {
                    str(i): self.env.action_spec().generate_value()
                    for i in range(n_agents)
                }
                multi_rewards = {}
                multi_discounts = {}
                next_multi_obs = {}

                # Simulate step for each agent - for simplicity, assume env is single-agent and we just replicate calls
                for i in range(n_agents):
                    # In a real multi-agent env, observations/rewards might be structured differently
                    # Here, we assume each agent interacts with a copy of the env or gets unique data
                    # For testing accumulator, it's fine to reuse self.env if state is managed per agent
                    # However, easiest is to generate random data per agent without full env stepping here
                    # Or, if using self.env, reset it per agent if their experiences are independent.
                    # For this test, let's assume independent experiences by resetting env or using distinct values.
                    # To keep it simple, we'll generate random values, not stepping a real env here for each agent.

                    # This part needs careful thought for true multi-agent env interaction.
                    # For testing the accumulator, we just need to ensure data gets pushed correctly.
                    # Let's use placeholder data based on current_obs_multi and multi_actions

                    # This simplified logic is for data generation, not true MARL interaction
                    multi_rewards[str(i)] = np.random.rand()
                    multi_discounts[str(i)] = 0.99
                    next_multi_obs[str(i)] = (
                        self.env.observation_spec().generate_value()
                    )  # New random obs for next step

                accumulator.push(
                    current_obs_multi,
                    multi_actions,
                    multi_rewards,
                    multi_discounts,
                    table_name=table_name,
                    next_state=next_multi_obs,
                )
                current_obs_multi = next_multi_obs

        self.assertTrue(
            accumulator.is_ready(batch_size),
            f"Multi-Agent RLPD Accumulator not ready. Status (online, offline): {accumulator.status}",
        )

        sample = accumulator.sample(batch_size)
        self.assertCountEqual(sample.keys(), self.signature_n1.keys())
        for k, v in sample.items():
            self.assertEqual(v.shape[0], batch_size)
            self.assertEqual(v.shape[1], n_multistep)

        online_size, offline_size = accumulator.status
        self.assertTrue(online_size >= batch_size // 2)
        self.assertTrue(offline_size >= batch_size // 2)

        if accumulator.server:
            accumulator.server.stop()


if __name__ == "__main__":
    absltest.main()
