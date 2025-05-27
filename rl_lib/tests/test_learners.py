# pylint: disable=g-import-not-at-top
# pylint: disable=protected-access
"""Tests for rl_lib.learners."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import haiku as hk
# import optax # F401: optax imported but unused
# import rlax  # F401: rlax imported but unused # Potentially for assertions or more complex test cases later
from typing import Dict, Any, List # F401: Tuple imported but unused

# Environment for testing
from bsuite.environments import catch

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

# Utilities
# from rl_lib.utils import build_signature # For Reverb/data signatures if needed in future tests.
# For now, defining NN in file, but could import from rl_lib.utils.NN


class ActionSpecMock:  # Simple mock for action_spec_per_sub_action
    def __init__(self, num_values: int):
        self.num_values = num_values
        self.shape = ()  # For discrete actions
        self.dtype = jnp.int32


class MockMultiActionManager(MultiActionManager):
    def __init__(
        self, action_spec_per_sub_action: ActionSpecMock, num_sub_actions: int
    ):
        super().__init__(
            action_spec=None
        )  # Base action_spec might not be directly used by MaskedMultiAction
        self.action_spec_per_sub_action = action_spec_per_sub_action
        self.num_sub_actions = num_sub_actions

    def iterate(self, nn_output: jax.Array) -> List[jax.Array]:
        # nn_output shape: (..., total_q_values_dim)
        # total_q_values_dim = num_sub_actions * self.action_spec_per_sub_action.num_values
        # Returns a list of JAX arrays, each for a sub_action's Q-values
        # Each element in list: (..., self.action_spec_per_sub_action.num_values)
        q_value_dim_per_action = self.action_spec_per_sub_action.num_values
        outputs = []
        # Assuming nn_output might have batch and/or sequence dimensions at the beginning
        # leading_dims = nn_output.shape[:-1] # F841: local variable 'leading_dims' is assigned to but never used
        for i in range(self.num_sub_actions):
            start_idx = i * q_value_dim_per_action
            end_idx = (i + 1) * q_value_dim_per_action
            # Slice only the last dimension
            sub_action_qs = nn_output[..., start_idx:end_idx]
            outputs.append(sub_action_qs)
        return outputs

    def get_mask(self, action_vector: jax.Array) -> jax.Array:
        # action_vector shape: (num_sub_actions,) - one chosen action index for each sub_action
        # Returns a mask of shape (num_sub_actions,)
        # This is a simplified mask for testing.
        # It should actually return a mask compatible with the output of iterate() for that sub-action.
        # For simplicity, let's make it return a scalar per sub-action, which can be broadcasted.
        # Or, more correctly, if iterate gives Qs of shape (num_choices), mask should be (num_choices)
        # The current MaskedMultiAction._loss expects a mask of shape (..., num_sub_actions)
        # which is then used as mask[..., i]. So this implementation should be fine.
        return jnp.where(action_vector > 0, 1.0, 0.5).astype(jnp.float32)


class SimpleMLP(hk.Module):
    """A simple MLP for testing QLearner."""

    def __init__(self, num_actions: int, name: str = "simple_mlp"):
        super().__init__(name=name)
        self._num_actions = num_actions

    def __call__(self, obs: jax.Array) -> jax.Array:
        network = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(50),
                jax.nn.relu,
                hk.Linear(self._num_actions),
            ]
        )
        return network(obs)


class TestLearners(parameterized.TestCase):
    """Tests for learner components."""

    def setUp(self):
        super().setUp()
        self.env = catch.Catch(seed=42)
        self.obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()

        # Haiku network transformation for QLearner
        def net_fn(obs: jax.Array) -> jax.Array:
            mlp = SimpleMLP(num_actions=self.action_spec.num_values)
            return mlp(obs)

        self.transformed_net = hk.without_apply_rng(hk.transform(net_fn))

        # For MaskedMultiAction
        self.num_sub_actions = 2
        self.sub_action_spec = ActionSpecMock(
            num_values=3
        )  # Each sub-action has 3 choices
        self.mam = MockMultiActionManager(self.sub_action_spec, self.num_sub_actions)

        def net_fn_masked(obs: jax.Array) -> jax.Array:
            mlp = SimpleMLP(
                num_actions=self.num_sub_actions * self.sub_action_spec.num_values
            )
            return mlp(obs)

        self.transformed_net_masked = hk.without_apply_rng(hk.transform(net_fn_masked))

        self.key = jax.random.PRNGKey(42)
        self.ema_decay = 0.99  # For EMATargetNet tests

    def _generate_dummy_observation(self) -> jax.Array:
        if hasattr(self.obs_spec, "generate_value") and callable(
            self.obs_spec.generate_value
        ):
            return self.obs_spec.generate_value()
        else:
            return jnp.zeros(self.obs_spec.shape, dtype=self.obs_spec.dtype)

    def _generate_sequence_batch(
        self, batch_size: int, sequence_length: int
    ) -> Dict[str, Any]:
        obs_shape = self.obs_spec.shape
        # Handle potentially complex/nested observation shapes
        if not isinstance(obs_shape, tuple):
            _dummy_obs = self._generate_dummy_observation()
            obs_shape = _dummy_obs.shape

        key_states, key_actions, key_rewards, key_next_states = jax.random.split(
            self.key, 4
        )
        self.key, _ = jax.random.split(
            self.key
        )  # Consume self.key to avoid reuse in subsequent calls

        states_data = [
            self._generate_dummy_observation()
            for _ in range(batch_size * sequence_length)
        ]
        states = jnp.array(states_data).reshape(
            (batch_size, sequence_length) + obs_shape
        )

        actions = jax.random.randint(
            key_actions,
            shape=(batch_size, sequence_length),
            minval=0,
            maxval=self.action_spec.num_values,
        )
        rewards = jax.random.uniform(key_rewards, shape=(batch_size, sequence_length))
        discounts = jnp.ones((batch_size, sequence_length), dtype=jnp.float32) * 0.99

        next_states_data = [
            self._generate_dummy_observation()
            for _ in range(batch_size * sequence_length)
        ]
        next_states = jnp.array(next_states_data).reshape(
            (batch_size, sequence_length) + obs_shape
        )

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

        # Replace actions with masked actions
        key_actions, self.key = jax.random.split(self.key)  # Get a new key
        actions_masked = jax.random.randint(
            key_actions,
            shape=(batch_size, sequence_length, self.num_sub_actions),
            minval=0,
            maxval=self.sub_action_spec.num_values,
            dtype=self.sub_action_spec.dtype,  # Ensure correct dtype for actions
        )
        batch["actions"] = actions_masked
        return batch

    def test_random_agent_actor_step(self):
        """Tests the actor_step of the Random agent."""
        agent = Random(action_spec=self.action_spec)

        dummy_params = {}  # Random agent doesn't use params
        key, actor_key = jax.random.split(self.key)

        observation = self._generate_dummy_observation()
        initial_actor_state = agent.initial_actor_state()

        action, new_actor_state = agent.actor_step(
            dummy_params, observation, initial_actor_state, actor_key, evaluation=False
        )

        self.assertIsInstance(action, jax.Array)
        self.assertEqual(action.shape, ())  # Scalar action
        self.assertTrue(0 <= action.item() < self.action_spec.num_values)
        self.assertEqual(
            new_actor_state, initial_actor_state
        )  # Random agent state doesn't change

    def test_qlearner_initial_train_state(self):
        """Tests the initial_train_state of the QLearner agent."""
        key, init_key = jax.random.split(self.key)
        agent = QLearner(
            network=self.transformed_net,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        train_state = agent.initial_train_state(init_key)

        self.assertIsInstance(train_state, dict)
        self.assertIn("params", train_state)
        self.assertIn("opt_state", train_state)
        self.assertNotEmpty(train_state["params"])  # Check that params are initialized

    def test_qlearner_actor_step(self):
        """Tests the actor_step of the QLearner agent."""
        key, init_key, actor_key_train, actor_key_eval = jax.random.split(self.key, 4)
        agent = QLearner(
            network=self.transformed_net,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        train_state = agent.initial_train_state(init_key)
        observation = self._generate_dummy_observation()
        initial_actor_state = agent.initial_actor_state()

        # Test actor_step in training mode
        action_train, new_actor_state_train = agent.actor_step(
            train_state,
            observation,
            initial_actor_state,
            actor_key_train,
            evaluation=False,
        )

        self.assertIsInstance(action_train, jax.Array)
        self.assertEqual(action_train.shape, ())
        self.assertTrue(0 <= action_train.item() < self.action_spec.num_values)
        self.assertEqual(new_actor_state_train, initial_actor_state)

        # Test actor_step in evaluation mode
        action_eval, new_actor_state_eval = agent.actor_step(
            train_state,
            observation,
            initial_actor_state,
            actor_key_eval,
            evaluation=True,
        )

        self.assertIsInstance(action_eval, jax.Array)
        self.assertEqual(action_eval.shape, ())
        self.assertTrue(0 <= action_eval.item() < self.action_spec.num_values)
        self.assertEqual(new_actor_state_eval, initial_actor_state)

    def test_qlearner_learner_step(self):
        """Tests the learner_step of the QLearner agent."""
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = QLearner(
            network=self.transformed_net,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]

        batch_size = 2
        # Create dummy batch data
        # Ensure observation spec shape is correctly handled
        obs_shape = self.obs_spec.shape
        if not isinstance(obs_shape, tuple):
            _dummy_obs = self._generate_dummy_observation()
            obs_shape = _dummy_obs.shape

        key_states, key_actions, key_rewards, key_next_states, self.key = (
            jax.random.split(self.key, 5)
        )

        dummy_states = jax.random.normal(
            key_states, (batch_size,) + obs_shape, dtype=self.obs_spec.dtype
        )
        dummy_actions = jax.random.randint(
            key_actions,
            (batch_size,),
            0,
            self.action_spec.num_values,
            dtype=self.action_spec.dtype,
        )
        dummy_rewards = jax.random.uniform(
            key_rewards, (batch_size,), dtype=jnp.float32
        )
        dummy_discounts = jnp.ones((batch_size,), dtype=jnp.float32) * 0.99
        dummy_next_states = jax.random.normal(
            key_next_states, (batch_size,) + obs_shape, dtype=self.obs_spec.dtype
        )

        batch = {
            "states": dummy_states,
            "actions": dummy_actions,
            "rewards": dummy_rewards,
            "discounts": dummy_discounts,
            "next_states": dummy_next_states,
        }

        new_train_state, loss = (
            agent.learner_step(  # Use agent._uncompiled_learner_step for direct testing if needed
                initial_train_state, batch, learner_key
            )
        )

        self.assertIsInstance(new_train_state, dict)
        self.assertIn("params", new_train_state)
        self.assertIn("opt_state", new_train_state)
        self.assertIsInstance(loss, jax.Array)
        self.assertEqual(loss.shape, ())  # Scalar loss

        # Check that parameters have changed (learning occurred)
        # Check that parameters have changed
        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal, original_params, new_train_state["params"]
            )
        )
        self.assertTrue(
            params_are_different, "Parameters did not change after learner_step."
        )

    def test_qlambda_learner_step(self):
        """Tests the learner_step of the QLambda agent with sequence data."""
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = QLambda(
            _lambda=0.7,
            network=self.transformed_net,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]

        batch = self._generate_sequence_batch(batch_size=2, sequence_length=3)

        new_train_state, loss = agent.learner_step(
            initial_train_state, batch, learner_key
        )

        self.assertIsInstance(new_train_state, dict)
        self.assertIn("params", new_train_state)
        self.assertIn("opt_state", new_train_state)
        self.assertIsInstance(loss, jax.Array)
        self.assertEqual(loss.shape, ())
        self.assertIsInstance(loss.item(), float)

        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal, original_params, new_train_state["params"]
            )
        )
        self.assertTrue(
            params_are_different,
            "QLambda parameters did not change after learner_step.",
        )

    def test_softwatkins_learner_step(self):
        """Tests the learner_step of the SoftWatkins agent with sequence data."""
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = SoftWatkins(
            kappa=0.5,
            _lambda=0.7,
            network=self.transformed_net,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]

        batch = self._generate_sequence_batch(batch_size=2, sequence_length=3)

        new_train_state, loss = agent.learner_step(
            initial_train_state, batch, learner_key
        )

        self.assertIsInstance(new_train_state, dict)
        self.assertIn("params", new_train_state)
        self.assertIn("opt_state", new_train_state)
        self.assertIsInstance(loss, jax.Array)
        self.assertEqual(loss.shape, ())
        self.assertIsInstance(loss.item(), float)

        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal, original_params, new_train_state["params"]
            )
        )
        self.assertTrue(
            params_are_different,
            "SoftWatkins parameters did not change after learner_step.",
        )

    def test_ematargetnet_initial_train_state(self):
        """Tests initial_train_state for EMATargetNet, ensuring ema_state is present and correct."""
        key, init_key = jax.random.split(self.key)
        agent = EMATargetNet(
            ema_decay=self.ema_decay,
            kappa=0.5,
            _lambda=0.7,
            network=self.transformed_net,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        train_state = agent.initial_train_state(init_key)

        self.assertIn("ema_state", train_state)
        # Check that initial params and ema_state (target params) are identical
        are_params_equal = jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal, train_state["params"], train_state["ema_state"]
            )
        )
        self.assertTrue(
            are_params_equal, "Initial params and ema_state should be identical."
        )

    def test_ematargetnet_learner_step_updates_ema(self):
        """Tests that EMATargetNet's learner_step updates ema_state correctly."""
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = EMATargetNet(
            ema_decay=self.ema_decay,  # Use self.ema_decay for the check later
            kappa=0.5,
            _lambda=0.7,
            network=self.transformed_net,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        initial_train_state = agent.initial_train_state(init_key)
        initial_params = initial_train_state["params"]
        initial_ema_state = initial_train_state["ema_state"]

        batch = self._generate_sequence_batch(batch_size=2, sequence_length=3)

        new_train_state, loss = agent.learner_step(
            initial_train_state, batch, learner_key
        )

        new_params = new_train_state["params"]
        new_ema_state = new_train_state["ema_state"]

        self.assertIsInstance(loss.item(), float)  # Basic check for loss

        # 1. Check that new_ema_state is different from initial_ema_state (unless params didn't change, which is unlikely)
        ema_states_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, initial_ema_state, new_ema_state)
        )
        # This check is only meaningful if params actually changed from initial_ema_state
        params_changed_from_initial_ema = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.array_equal, initial_ema_state, new_params)
        )
        if params_changed_from_initial_ema:
            self.assertTrue(
                ema_states_are_different,
                "EMA state should change if online params change from initial EMA state.",
            )
        else:
            # If new_params are somehow identical to initial_ema_state, then new_ema_state should also be identical.
            self.assertTrue(
                jax.tree_util.tree_all(
                    jax.tree_util.tree_map(
                        jnp.array_equal, new_ema_state, initial_ema_state
                    )
                ),
                "EMA state should not change if new params are identical to initial EMA state.",
            )

        # 2. More precise check: new_ema_state should be an EMA of initial_ema_state and new_params
        # Note: The EMATargetNet updates ema_state using the *newly updated* online parameters.
        # So, the check should be: new_ema_state = decay * initial_ema_state + (1-decay) * new_params
        # (assuming ema_fn in EMATargetNet is called with state['ema_state'] (old) and updated_state['params'] (new))

        expected_ema_state_leaves = jax.tree_util.tree_map(
            lambda old_ema, p_new: self.ema_decay * old_ema
            + (1.0 - self.ema_decay) * p_new,
            initial_ema_state,  # This is state['ema_state'] before the learner_step's super call returns
            new_params,  # This is updated_state_base['params'] after super call returns
        )

        are_ema_updated_correctly = jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.allclose, new_ema_state, expected_ema_state_leaves
            )
        )
        self.assertTrue(
            are_ema_updated_correctly,
            "EMA state was not updated correctly according to the EMA rule.",
        )

    def test_maskedmulti_initial_train_state(self):
        """Tests initial_train_state for MaskedMultiAction."""
        key, init_key = jax.random.split(self.key)
        agent = MaskedMultiAction(
            mam=self.mam,
            ema_decay=self.ema_decay,
            kappa=0.5,
            _lambda=0.7,
            network=self.transformed_net_masked,  # Use the network with correct output dim
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        train_state = agent.initial_train_state(init_key)

        self.assertIn("params", train_state)
        self.assertIn("opt_state", train_state)
        self.assertIn("ema_state", train_state)
        are_params_equal = jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal, train_state["params"], train_state["ema_state"]
            )
        )
        self.assertTrue(
            are_params_equal,
            "Initial params and ema_state should be identical for MaskedMultiAction.",
        )

    def test_maskedmulti_actor_step(self):
        """Tests actor_step for MaskedMultiAction."""
        key, init_key, actor_key = jax.random.split(self.key, 3)
        agent = MaskedMultiAction(
            mam=self.mam,
            ema_decay=self.ema_decay,
            kappa=0.5,
            _lambda=0.7,
            network=self.transformed_net_masked,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        train_state = agent.initial_train_state(init_key)
        observation = self._generate_dummy_observation()
        initial_actor_state = (
            agent.initial_actor_state()
        )  # Should be empty tuple from QLearner

        chosen_actions, new_actor_state = agent.actor_step(
            train_state, observation, initial_actor_state, actor_key, evaluation=False
        )

        self.assertIsInstance(chosen_actions, list)
        self.assertLen(chosen_actions, self.num_sub_actions)
        for sub_action in chosen_actions:
            self.assertIsInstance(sub_action, jax.Array)
            self.assertEqual(sub_action.shape, ())  # Scalar action for each sub_action
            self.assertTrue(0 <= sub_action.item() < self.sub_action_spec.num_values)
        self.assertEqual(new_actor_state, initial_actor_state)

    def test_maskedmulti_learner_step(self):
        """Tests learner_step for MaskedMultiAction."""
        key, init_key, learner_key = jax.random.split(self.key, 3)
        agent = MaskedMultiAction(
            mam=self.mam,
            ema_decay=self.ema_decay,
            kappa=0.5,
            _lambda=0.7,
            network=self.transformed_net_masked,
            observation_spec=self.obs_spec,
            learning_rate=0.001,
        )

        initial_train_state = agent.initial_train_state(init_key)
        original_params = initial_train_state["params"]
        initial_ema_state = initial_train_state["ema_state"]

        batch = self._generate_masked_sequence_batch(batch_size=2, sequence_length=3)

        new_train_state, loss = agent.learner_step(
            initial_train_state, batch, learner_key
        )

        self.assertIsInstance(new_train_state, dict)
        self.assertIn("params", new_train_state)
        self.assertIn("opt_state", new_train_state)
        self.assertIn("ema_state", new_train_state)
        self.assertIsInstance(loss, jax.Array)
        self.assertEqual(loss.shape, ())
        self.assertIsInstance(loss.item(), float)

        # Check params changed
        params_are_different = not jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                jnp.array_equal, original_params, new_train_state["params"]
            )
        )
        self.assertTrue(
            params_are_different,
            "MaskedMultiAction parameters did not change after learner_step.",
        )

        # Check EMA update
        new_params = new_train_state["params"]
        new_ema_state = new_train_state["ema_state"]
        expected_ema_state = jax.tree_util.tree_map(
            lambda old_ema, p_new: self.ema_decay * old_ema
            + (1.0 - self.ema_decay) * p_new,
            initial_ema_state,
            new_params,
        )
        are_ema_updated_correctly = jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.allclose, new_ema_state, expected_ema_state)
        )
        self.assertTrue(
            are_ema_updated_correctly,
            "MaskedMultiAction EMA state was not updated correctly.",
        )


if __name__ == "__main__":
    absltest.main()
