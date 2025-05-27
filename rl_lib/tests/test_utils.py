from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf
import jax
import jax.numpy as jnp
import flax.linen as nn # For testing NN
from jax.tree_util import tree_leaves


from rl_lib.utils import EMATree, NN as UtilsNN, build_signature, cost, l2_loss, l2_regulariser # Renamed imported NN to UtilsNN


class MockSpec: # Kept for build_signature test as it uses tf.TensorSpec-like objects
    def __init__(self, shape, dtype, name=None):
        self.shape = shape
        self.dtype = dtype # Should be tf.dtype for build_signature
        self.name = name


class TestUtils(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(0)

    def test_ema_constant_value(self):
        ema_decay = 0.5
        ema = EMATree(ema_decay)
        
        initial_val_dict = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        current_ema_state = ema.init(initial_val_dict) # Initial state is the value itself

        # Update with the same value
        new_value_dict = initial_val_dict
        updated_ema_state = ema(current_ema_state, new_value_dict)

        # Expected: 0.5 * 1.0 + 0.5 * 1.0 = 1.0 for 'a'
        # Expected: 0.5 * 2.0 + 0.5 * 2.0 = 2.0 for 'b'
        self.assertTrue(jnp.allclose(updated_ema_state["a"], 1.0))
        self.assertTrue(jnp.allclose(updated_ema_state["b"], 2.0))

    def test_ema_changing_value(self):
        ema_decay = 0.9
        ema = EMATree(ema_decay)

        initial_value = {"val": jnp.array(10.0)}
        current_ema_state = ema.init(initial_value)

        new_value_1 = {"val": jnp.array(20.0)}
        current_ema_state = ema(current_ema_state, new_value_1)
        expected_val_1 = ema_decay * 10.0 + (1.0 - ema_decay) * 20.0 # 0.9*10 + 0.1*20 = 9 + 2 = 11.0
        self.assertTrue(jnp.allclose(current_ema_state["val"], expected_val_1))

        new_value_2 = {"val": jnp.array(0.0)}
        current_ema_state = ema(current_ema_state, new_value_2)
        expected_val_2 = ema_decay * expected_val_1 + (1.0 - ema_decay) * 0.0 # 0.9 * 11.0 = 9.9
        self.assertTrue(jnp.allclose(current_ema_state["val"], expected_val_2))


    def test_build_signature(self):
        # Using tf.TensorSpec directly for obs_spec and action_spec as build_signature expects
        obs_spec = tf.TensorSpec(shape=(84, 84, 3), dtype=tf.uint8)
        action_spec = tf.TensorSpec(shape=(), dtype=tf.int32)
        n_multistep = 4

        signature = build_signature(obs_spec, action_spec, n_multistep)

        self.assertIsInstance(signature, dict)
        expected_keys = ["states", "actions", "rewards", "discounts", "next_states"]
        self.assertCountEqual(signature.keys(), expected_keys)

        for key in expected_keys:
            self.assertIsInstance(signature[key], tf.TensorSpec)

        self.assertEqual(list(signature["states"].shape), [n_multistep, 84, 84, 3])
        self.assertEqual(list(signature["actions"].shape), [n_multistep])
        self.assertEqual(list(signature["rewards"].shape), [n_multistep])
        self.assertEqual(list(signature["discounts"].shape), [n_multistep])
        self.assertEqual(list(signature["next_states"].shape), [n_multistep, 84, 84, 3])

        self.assertEqual(signature["states"].dtype, obs_spec.dtype)
        self.assertEqual(signature["actions"].dtype, action_spec.dtype)
        self.assertEqual(signature["rewards"].dtype, tf.float32)
        self.assertEqual(signature["discounts"].dtype, tf.float32)
        self.assertEqual(signature["next_states"].dtype, obs_spec.dtype)


    def test_cost_function(self):
        error1 = jnp.array([1.0, 2.0, 3.0])
        error2 = jnp.array([-1.0, -2.0, -3.0])
        error0 = jnp.array([0.0, 0.0, 0.0])

        cost_val0 = cost(error0)
        cost_val1 = cost(error1)
        cost_val2 = cost(error2)

        self.assertTrue(jnp.allclose(cost_val0, 0.0))
        self.assertTrue(cost_val1 > 0)
        self.assertTrue(jnp.allclose(cost_val1, cost_val2))

        expected_manual_cost = (
            jnp.mean(jnp.abs(error1))
            + jnp.mean(jnp.square(error1))
            + jnp.mean(jnp.abs(error1) ** 3)
        ) / 3.0
        self.assertTrue(jnp.allclose(cost_val1, expected_manual_cost))

    def test_l2_loss(self):
        params = jnp.array([1.0, 2.0, 3.0])
        alpha = 0.1
        loss_val = l2_loss(params, alpha) # Renamed variable to avoid conflict
        expected_loss = 0.1 * ((1**2 + 2**2 + 3**2) / 3.0)
        self.assertTrue(jnp.allclose(loss_val, expected_loss))

    def test_l2_regulariser(self):
        params_tree = {
            "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.1, 0.2])},
            "layer2": {"w": jnp.array([-1.0, -2.0])},
            "empty_leaf": {}, # Test robustness to empty or non-array leaves
            "none_leaf": None,
        }
        alpha = 0.1
        reg_loss = l2_regulariser(params_tree, alpha=alpha)

        expected_reg_loss_val = 0.0
        for leaf in tree_leaves(params_tree):
            if isinstance(leaf, jax.Array) and leaf.size > 0: # l2_regulariser filters Nones and non-arrays
                expected_reg_loss_val += alpha * jnp.mean(jnp.square(leaf))
        
        self.assertTrue(jnp.allclose(reg_loss, expected_reg_loss_val))

    def test_nn_forward_pass_flax(self):
        n_dense = 3
        width = 64
        output_dims = 10
        key_init, key_apply = jax.random.split(self.key)

        # Use the NN from rl_lib.utils
        flax_net = UtilsNN(n_dense=n_dense, width=width, output_dims=output_dims)

        dummy_obs_single = jnp.ones((32,))
        dummy_obs_batch = jnp.ones((5, 32))

        # Initialize parameters
        params = flax_net.init(key_init, dummy_obs_single)['params']

        # Test with single observation
        output_single = flax_net.apply({'params': params}, dummy_obs_single)
        self.assertEqual(output_single.shape, (output_dims,))

        # Test with a batch of observations
        # Flax modules typically handle batching automatically if input has a batch dimension.
        # The NN in utils flattens input `jnp.reshape(x, -1)`.
        # For batched input (B, Features), Dense layers will produce (B, output_features).
        # The reshape to -1 on (B, Features) will make it (B * Features), which is not what we want for batch processing.
        # The NN in utils.py should be vmapped or its internal reshape handled carefully for batching.
        # The current UtilsNN is designed for single instance processing due to `jnp.reshape(x, -1)`.
        # To test batch processing, we should vmap it.

        # Test vmapped application for batch
        vmapped_apply = jax.vmap(flax_net.apply, in_axes=({'params': None}, 0))
        output_batch_vmapped = vmapped_apply({'params': params}, dummy_obs_batch)
        self.assertEqual(output_batch_vmapped.shape, (dummy_obs_batch.shape[0], output_dims))

        # If the NN was intended to process batches directly (e.g. by reshaping (B, *F) to (B, prod(F))):
        # The current UtilsNN:
        #   `processed_x: JaxArray = jnp.reshape(x, -1)`
        #   If x is (B, F1, F2), this becomes (B*F1*F2,).
        #   The Dense layers then operate on this flattened version.
        # This means the provided UtilsNN is not batch-aware by default in its architecture.
        # The test should reflect how it's used by learners (likely with vmap or single instances).
        # The QLearner uses `jax.vmap(network.apply, in_axes=(None, 0))` if network.apply takes (params, obs).
        # Our Flax network.apply takes ({'params': params}, obs).
        # The QLearner's `_batch_apply_net` correctly vmaps a helper.
        # So, the vmapped test is more representative.


if __name__ == "__main__":
    absltest.main()