from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf
import jax
import jax.numpy as jnp
import haiku as hk
from jax.tree_util import tree_leaves


from rl_lib.utils import EMATree, NN, build_signature, cost, l2_loss, l2_regulariser


class MockSpec:
    def __init__(self, shape, dtype, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name


class TestUtils(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(0)

    def test_ema(self, k=5):
        ema = EMATree(0.5)
        # rng = hk.PRNGSequence(jax.random.PRNGKey(0)) # Not used in this version of test_ema

        def get_val():
            return {"a": jnp.ones(()), "b": jnp.ones(())}

        x = get_val()
        state = ema.init(x)

        for _ in range(k):
            # x = get_val() # x is not reassigned, so state will be EMA of initial x and itself.
            state = ema(state, x)  # Corrected: ema(current_ema_state, new_value)

        # With ema_decay = 0.5, state = 0.5 * state + 0.5 * x
        # If x is always {'a': 1.0, 'b': 1.0} and state starts as x:
        # iter 1: state = 0.5*1 + 0.5*1 = 1
        # iter 2: state = 0.5*1 + 0.5*1 = 1
        # ... so it remains 1.0.
        self.assertEqual(state["a"], 1.0)

    def test_build_signature(self):
        obs_spec = MockSpec(shape=(84, 84, 3), dtype=jnp.uint8)
        action_spec = MockSpec(shape=(), dtype=jnp.int32)  # Scalar action
        n_multistep = 4

        signature = build_signature(obs_spec, action_spec, n_multistep)

        self.assertIsInstance(signature, dict)
        expected_keys = ["states", "actions", "rewards", "discounts", "next_states"]
        self.assertCountEqual(signature.keys(), expected_keys)

        for key in expected_keys:
            self.assertIsInstance(signature[key], tf.TensorSpec)

        self.assertEqual(signature["states"].shape.as_list(), [n_multistep, 84, 84, 3])
        self.assertEqual(
            signature["actions"].shape.as_list(), [n_multistep]
        )  # Scalar action, only n_multistep
        self.assertEqual(signature["rewards"].shape.as_list(), [n_multistep])
        self.assertEqual(signature["discounts"].shape.as_list(), [n_multistep])
        self.assertEqual(
            signature["next_states"].shape.as_list(), [n_multistep, 84, 84, 3]
        )

        self.assertEqual(signature["states"].dtype, tf.dtypes.as_dtype(obs_spec.dtype))
        self.assertEqual(
            signature["actions"].dtype, tf.dtypes.as_dtype(action_spec.dtype)
        )
        self.assertEqual(signature["rewards"].dtype, tf.float32)
        self.assertEqual(signature["discounts"].dtype, tf.float32)
        self.assertEqual(
            signature["next_states"].dtype, tf.dtypes.as_dtype(obs_spec.dtype)
        )

    def test_cost_function(self):
        error1 = jnp.array([1.0, 2.0, 3.0])
        error2 = jnp.array([-1.0, -2.0, -3.0])
        error0 = jnp.array([0.0, 0.0, 0.0])

        cost_val0 = cost(error0)
        cost_val1 = cost(error1)
        cost_val2 = cost(error2)

        self.assertEqual(cost_val0, 0.0)
        self.assertTrue(cost_val1 > 0)
        self.assertEqual(cost_val1, cost_val2)

        # Manual calculation for error1
        # abs_e1 = [1, 2, 3]
        # mean_abs_e1 = (1+2+3)/3 = 2
        # abs_e1_sq = [1, 4, 9]
        # mean_abs_e1_sq = (1+4+9)/3 = 14/3
        # abs_e1_cub = [1, 8, 27]
        # mean_abs_e1_cub = (1+8+27)/3 = 36/3 = 12
        # expected_manual_cost = (2 + 14/3 + 12) / 3 = (6/3 + 14/3 + 36/3) / 3 = (56/3) / 3 = 56/9
        expected_manual_cost = (
            jnp.mean(jnp.abs(error1))
            + jnp.mean(jnp.square(error1))
            + jnp.mean(jnp.abs(error1) ** 3)
        ) / 3.0
        self.assertTrue(jnp.allclose(cost_val1, expected_manual_cost))

    def test_l2_loss(self):
        params = jnp.array([1.0, 2.0, 3.0])
        alpha = 0.1
        loss = l2_loss(params, alpha)
        expected_loss = 0.1 * ((1**2 + 2**2 + 3**2) / 3.0)  # 0.1 * (14/3)
        self.assertTrue(jnp.allclose(loss, expected_loss))

    def test_l2_regulariser(self):
        params_tree = {
            "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.1, 0.2])},
            "layer2": {"w": jnp.array([-1.0, -2.0])},
        }
        alpha = 0.1
        reg_loss = l2_regulariser(params_tree, alpha=alpha)

        expected_reg_loss = 0.0
        for leaf in tree_leaves(params_tree):
            expected_reg_loss += alpha * jnp.mean(jnp.square(leaf))

        self.assertTrue(jnp.allclose(reg_loss, expected_reg_loss))

    def test_nn_forward_pass(self):
        n_dense = 3
        width = 64
        output_dims = 10

        def net_fn(obs):
            net = NN(n_dense=n_dense, width=width, output_dims=output_dims)
            return net(obs)

        transformed_net = hk.without_apply_rng(hk.transform(net_fn))

        # Example observation shape (batch_size, feature_dim)
        # NN flattens input, so actual feature dim structure doesn't matter as much
        # as long as it can be passed to hk.Linear
        dummy_obs_single = jnp.ones((32,))
        dummy_obs_batch = jnp.ones((5, 32))  # Batch of 5 observations

        key, init_key = jax.random.split(self.key)
        params = transformed_net.init(init_key, dummy_obs_single)

        # Test with single observation
        output_single = transformed_net.apply(params, dummy_obs_single)
        self.assertEqual(output_single.shape, (output_dims,))

        # Test with a batch of observations (hk.transform handles vmap implicitly for apply)
        # However, our NN is defined to take a single instance and flatten it.
        # If we want batch processing, the NN's __call__ or the way we apply it needs to be vmapped.
        # The current NN definition `x = jnp.reshape(x, -1)` implies it processes one item.
        # Let's test it as defined: one item at a time.
        # If batch support is inherent in hk.Module's Dense layers when input has a batch dim,
        # then (batch_size, output_dims) would be the shape.
        # hk.transform + hk.Sequential + hk.Linear typically handles batch dimension automatically.

        output_batch = transformed_net.apply(params, dummy_obs_batch)
        self.assertEqual(output_batch.shape, (dummy_obs_batch.shape[0], output_dims))


if __name__ == "__main__":
    absltest.main()
