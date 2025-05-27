# pylint: disable=g-import-not-at-top
# pylint: disable=protected-access
"""Tests for rl_lib.learners."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import flax.linen as nn # Changed from haiku
import numpy as np
from typing import Dict, Any, List

# Environment for testing
import gymnasium as gym

# Learners to test
from rl_lib.learners import (
    Random,
    QLearner,
    QLambda,
    SoftWatkins,
    EMATargetNet,
    MultiActionManager,
    MaskedMultiAction,
)

# Helper Spec classes for Gymnasium
class GymObservationSpec:
    def __init__(self, observation_space: gym.spaces.Space): # More general space
        self.shape = observation_space.shape
        self.dtype = observation_space.dtype if hasattr(observation_space, 'dtype') else jnp.float32
        if not self.shape: # Handle Discrete observation spaces if they occur
             if hasattr(observation_space, 'n'):
                 self.shape = (1,) # Represent as a 1-element vector
                 self.dtype = jnp.int32


    def generate_value(self):
        # Create a JAX array of zeros with the correct shape and dtype
        return jnp.zeros(self.shape, dtype=self.dtype)

class GymActionSpec:
    def __init__(self, action_space: gym.spaces.Discrete):
        self.shape = () # Scalar for discrete actions
        self.dtype = jnp.int32
        self.num_values = int(action_space.n)

    def generate_value(self):
        return jnp.array(0, dtype=self.dtype)


class ActionSpecMockForMulti:  # Simple mock for action_spec_per_sub_action
    def __init__(self, num_values: int):
        self.num_values = num_values
        self.shape = ()
        self.dtype = jnp.int32


class MockMultiActionManager(MultiActionManager):
    def __init__(
        self, action_spec_per_sub_action: ActionSpecMockForMulti, num_sub_actions: int
    ):
        # The base MultiActionManager action_spec is not strictly typed or used by MaskedMultiAction directly
        # It's more of a placeholder for the composite action's properties if needed.
        super().__init__(action_spec=None)
        self.action_spec_per_sub_action = action_spec_per_sub_action
        self.num_sub_actions = num_sub_actions

    def iterate(self, nn_output: jax.Array) -> List[jax.Array]:
        q_value_dim_per_action = self.action_spec_per_sub_action.num_values
        outputs = []
        for i in range(self.num_sub_actions):
            start_idx = i * q_value_dim_per_action
            end_idx = (i + 1) * q_value_dim_per_action
            sub_action_qs = nn_output[..., start_idx:end_idx]
            outputs.append(sub_action_qs)
        return outputs

    def get_mask(self, action_from_one_head: jax.Array) -> jax.Array:
        # MaskedMultiAction._loss applies this mask to the TD error of a specific head.
        # The input `action_from_one_head` is the action taken for that head.
        # The mask should be broadcastable with the TD error (scalar or same shape as Q-values for that head).
        # For simplicity, returning a scalar mask.
        # Original: jnp.where(action_vector > 0, 1.0, 0.5) - action_vector was (num_sub_actions,)
        # Here, action_from_one_head is scalar.
        return jnp.array(1.0, dtype=jnp.float32) # Always allow learning for this head for testing


class SimpleFlaxMLP(nn.Module):
    """A simple MLP for testing QLearner, using Flax."""
    num_actions: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        obs_flat = obs.reshape((obs.shape[0], -1)) if obs.ndim > 1 else obs.reshape((1, -1)) # Handle single/batch
        x = nn.Dense(features=50)(obs_flat)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_actions)(x)
        return x.squeeze() # Squeeze if single obs was passed and reshaped


class TestLearners(parameterized.TestCase):
    """Tests for learner components."""

    def setUp(self):
        super().setUp()
        self.env = gym.make("CartPole-v1")
        self.obs_spec = GymObservationSpec(self.env.observation_space)
        self.action_spec = GymActionSpec(self.env.action_space)

        self.flax_mlp = SimpleFlaxMLP(num_actions=self.action_spec.num_values)

        self.num_sub_actions = 2
        self.sub_action_spec = ActionSpecMockForMulti(num_values=3)
        self.mam = MockMultiActionManager(self.sub_action_spec, self.num_sub_actions)
        self.flax_mlp_masked = SimpleFlaxMLP(
            num_actions=self.num_sub_actions * self.sub_action_spec.num_values
        )

        self.key = jax.random.PRNGKey(42)
        self.ema_decay = 0.99

    def _generate_dummy_observation(self) -> jax.Array:
        return self.obs_spec.generate_value()

    def _generate_sequence_batch(
        self, batch_size: int, sequence_length: int
    ) -> Dict[str, Any]:
        obs_shape = self.obs_spec.shape
        key_states, key_actions, key_rewards, key_next_states, self.key = jax.random.split(self.key, 5)

        states = jax.random.normal(key_states, (batch_size, sequence_length) + obs_shape, dtype=self.obs_spec.dtype)
        actions = jax.random.randint(
            key_actions, (batch_size, sequence_length), 0, self.action_spec.num_values, dtype=self.action_spec.dtype
        )
        rewards = jax.random.uniform(key_rewards, (batch_size, sequence_length))
        discounts = jnp.ones((batch_size, sequence_length), dtype=jnp.float32) * 0.99
        next_states = jax.random.normal(key_next_states, (batch_size, sequence_length) + obs_shape, dtype=self.obs_spec.dtype)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "discounts": discounts,
            "next_states": next_states,
        }

    def _generate_masked_sequence_batch(
        self, batch_size: int, sequence_length: int
    ) -> Dict[str, Any]:
        batch = self._generate_sequence_batch(batch_size, sequence_length)
        key_actions, self.key = jax.random.split(self.key)
        actions_masked = jax.random.randint(
            key_actions,
            (batch_size, sequence_length, self.num_sub_actions),
            0, self.sub_action_spec.num_values, dtype=self.sub_action_spec.dtype,
        )
        batch["actions"] = actions_masked
        return batch

    def test_random_agent_actor_step(self):
        agent = Random(action_spec=self.action_spec)
        key, actor_key = jax.random.split(self.key)
        observation = self._generate_dummy_observation()
        initial_actor_state = agent.initial_actor_state()
        # Random agent's train_state is an empty dict, not params
        train_state = agent.initial_train_state(key)


        action, new_actor_state = agent.actor_step(
            train_state, observation, initial_actor_state, actor_key, evaluation=False
        )

        self.assertIsInstance(action, jax.Array)
        self.assertEqual(action.shape, ())
        self.assertTrue(0 <= action.item() < self.action_spec.num_values)
        self.assertEqual(new_actor_state, initial_actor_state)

    def test_qlearner_initial_train_state(self):
        key, init_key = jax.random.split(self.key)
        agent = QLearner(
            network=self.flax_mlp, # Pass Flax module instance
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )
        train_state = agent.initial_train_state(init_key)
        self.assertIsInstance(train_state, dict)
        self.assertIn("params", train_state)
        self.assertIn("opt_state", train_state)
        self.assertNotEmpty(train_state["params"])

    def test_qlearner_actor_step(self):
        key, init_key, actor_key_train, actor_key_eval = jax.random.split(self.key, 4)
        agent = QLearner(
            network=self.flax_mlp,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )
        train_state = agent.initial_train_state(init_key)
        observation = self._generate_dummy_observation()
        initial_actor_state = agent.initial_actor_state()

        action_train, new_actor_state_train = agent.actor_step(
            train_state, observation, initial_actor_state, actor_key_train, evaluation=False,
        )
        self.assertIsInstance(action_train, jax.Array)
        self.assertEqual(action_train.shape, ())
        self.assertTrue(0 <= action_train.item() < self.action_spec.num_values)
        self.assertEqual(new_actor_state_train, initial_actor_state)

        action_eval, new_actor_state_eval = agent.actor_step(
            train_state, observation, initial_actor_state, actor_key_eval, evaluation=True,
        )
        self.assertIsInstance(action_eval, jax.Array)
        self.assertEqual(action_eval.shape, ())
        self.assertTrue(0 <= action_eval.item() < self.action_spec.num_values)
        self.assertEqual(new_actor_state_eval, initial_actor_state)

    def test_qlearner_learner_step(self):
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = QLearner(
            network=self.flax_mlp,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )
        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]

        batch_size = 2
        obs_shape = self.obs_spec.shape
        key_states, key_actions, key_rewards, key_next_states, self.key = jax.random.split(self.key, 5)

        dummy_states = jax.random.normal(key_states, (batch_size,) + obs_shape, dtype=self.obs_spec.dtype)
        dummy_actions = jax.random.randint(key_actions, (batch_size,), 0, self.action_spec.num_values, dtype=self.action_spec.dtype)
        dummy_rewards = jax.random.uniform(key_rewards, (batch_size,), dtype=jnp.float32)
        dummy_discounts = jnp.ones((batch_size,), dtype=jnp.float32) * 0.99
        dummy_next_states = jax.random.normal(key_next_states, (batch_size,) + obs_shape, dtype=self.obs_spec.dtype)

        batch = {
            "states": dummy_states, "actions": dummy_actions, "rewards": dummy_rewards,
            "discounts": dummy_discounts, "next_states": dummy_next_states,
        }
        new_train_state, loss = agent.learner_step(initial_train_state, batch, learner_key)

        self.assertIsInstance(new_train_state, dict)
        self.assertIn("params", new_train_state)
        self.assertIn("opt_state", new_train_state)
        self.assertIsInstance(loss, jax.Array)
        self.assertEqual(loss.shape, ())

        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, original_params, new_train_state["params"])
        )
        self.assertTrue(params_are_different, "Parameters did not change after learner_step.")

    def test_qlambda_learner_step(self):
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = QLambda(
            _lambda=0.7, network=self.flax_mlp, observation_spec=self.obs_spec, learning_rate=0.001,
        )
        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]
        batch = self._generate_sequence_batch(batch_size=2, sequence_length=3)
        new_train_state, loss = agent.learner_step(initial_train_state, batch, learner_key)

        self.assertIsInstance(new_train_state, dict)
        self.assertIsInstance(loss.item(), float)
        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, original_params, new_train_state["params"])
        )
        self.assertTrue(params_are_different, "QLambda parameters did not change.")

    def test_softwatkins_learner_step(self):
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = SoftWatkins(
            kappa=0.5, _lambda=0.7, network=self.flax_mlp, observation_spec=self.obs_spec, learning_rate=0.001,
        )
        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]
        batch = self._generate_sequence_batch(batch_size=2, sequence_length=3)
        new_train_state, loss = agent.learner_step(initial_train_state, batch, learner_key)

        self.assertIsInstance(new_train_state, dict)
        self.assertIsInstance(loss.item(), float)
        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, original_params, new_train_state["params"])
        )
        self.assertTrue(params_are_different, "SoftWatkins parameters did not change.")

    def test_ematargetnet_initial_train_state(self):
        key, init_key = jax.random.split(self.key)
        agent = EMATargetNet(
            ema_decay=self.ema_decay, kappa=0.5, _lambda=0.7, network=self.flax_mlp,
            observation_spec=self.obs_spec, learning_rate=0.001,
        )
        train_state = agent.initial_train_state(init_key)
        self.assertIn("ema_target_params", train_state) # Changed from ema_state to ema_target_params
        are_params_equal = jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, train_state["params"], train_state["ema_target_params"])
        )
        self.assertTrue(are_params_equal, "Initial params and ema_target_params should be identical.")

    def test_ematargetnet_learner_step_updates_ema(self):
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = EMATargetNet(
            ema_decay=self.ema_decay, kappa=0.5, _lambda=0.7, network=self.flax_mlp,
            observation_spec=self.obs_spec, learning_rate=0.001,
        )
        initial_train_state = agent.initial_train_state(init_key)
        # initial_params = initial_train_state["params"] # Not directly used for expected EMA calc here
        initial_target_params = initial_train_state["ema_target_params"]
        batch = self._generate_sequence_batch(batch_size=2, sequence_length=3)
        new_train_state, loss = agent.learner_step(initial_train_state, batch, learner_key)

        new_online_params = new_train_state["params"]
        new_target_params = new_train_state["ema_target_params"]
        self.assertIsInstance(loss.item(), float)

        # Check if new_target_params are different from initial_target_params
        # This is only meaningful if new_online_params are different from initial_target_params
        target_params_changed = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, initial_target_params, new_target_params)
        )
        online_params_changed_from_initial_target = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, initial_target_params, new_online_params)
        )

        if online_params_changed_from_initial_target:
            self.assertTrue(target_params_changed, "Target params should change if online params changed from them.")
        else:
             self.assertFalse(target_params_changed, "Target params should not change if online params are same as initial target.")


        # Expected EMA: new_target = decay * old_target + (1-decay) * new_online
        expected_target_params_tree = jax.tree_util.tree_map(
            lambda old_target, p_new_online: self.ema_decay * old_target + (1.0 - self.ema_decay) * p_new_online,
            initial_target_params, # old target params from before this step
            new_online_params,     # new online params from this step
        )
        are_ema_updated_correctly = jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.allclose, new_target_params, expected_target_params_tree)
        )
        self.assertTrue(are_ema_updated_correctly, "EMA target params not updated correctly.")

    def test_maskedmulti_initial_train_state(self):
        key, init_key = jax.random.split(self.key)
        agent = MaskedMultiAction(
            mam=self.mam, ema_decay=self.ema_decay, kappa=0.5, _lambda=0.7,
            network=self.flax_mlp_masked, observation_spec=self.obs_spec, learning_rate=0.001,
        )
        train_state = agent.initial_train_state(init_key)
        self.assertIn("params", train_state)
        self.assertIn("opt_state", train_state)
        self.assertIn("ema_target_params", train_state)
        are_params_equal = jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, train_state["params"], train_state["ema_target_params"])
        )
        self.assertTrue(are_params_equal, "Initial params and ema_target_params should be identical.")

    def test_maskedmulti_actor_step(self):
        key, init_key, actor_key = jax.random.split(self.key, 3)
        agent = MaskedMultiAction(
            mam=self.mam, ema_decay=self.ema_decay, kappa=0.5, _lambda=0.7,
            network=self.flax_mlp_masked, observation_spec=self.obs_spec, learning_rate=0.001,
        )
        train_state = agent.initial_train_state(init_key)
        observation = self._generate_dummy_observation()
        initial_actor_state = agent.initial_actor_state()

        chosen_actions, new_actor_state = agent.actor_step(
            train_state, observation, initial_actor_state, actor_key, evaluation=False
        )
        self.assertIsInstance(chosen_actions, list)
        self.assertLen(chosen_actions, self.num_sub_actions)
        for sub_action in chosen_actions:
            self.assertIsInstance(sub_action, jax.Array)
            self.assertEqual(sub_action.shape, ())
            self.assertTrue(0 <= sub_action.item() < self.sub_action_spec.num_values)
        self.assertEqual(new_actor_state, initial_actor_state)

    def test_maskedmulti_learner_step(self):
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = MaskedMultiAction(
            mam=self.mam, ema_decay=self.ema_decay, kappa=0.5, _lambda=0.7,
            network=self.flax_mlp_masked, observation_spec=self.obs_spec, learning_rate=0.001,
        )
        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]
        initial_target_params = initial_train_state["ema_target_params"]
        batch = self._generate_masked_sequence_batch(batch_size=2, sequence_length=3)
        new_train_state, loss = agent.learner_step(initial_train_state, batch, learner_key)

        self.assertIsInstance(new_train_state, dict)
        self.assertIsInstance(loss.item(), float)
        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, original_params, new_train_state["params"])
        )
        self.assertTrue(params_are_different, "MaskedMultiAction parameters did not change.")

        new_online_params = new_train_state["params"]
        new_target_params = new_train_state["ema_target_params"]
        expected_target_params = jax.tree_util.tree_map(
            lambda old_target, p_new_online: self.ema_decay * old_target + (1.0 - self.ema_decay) * p_new_online,
            initial_target_params, new_online_params,
        )
        are_ema_updated_correctly = jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.allclose, new_target_params, expected_target_params)
        )
        self.assertTrue(are_ema_updated_correctly, "MaskedMultiAction EMA state not updated correctly.")


if __name__ == "__main__":
    absltest.main()