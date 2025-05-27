# pylint: disable=g-import-not-at-top
"""Tests for rl_lib.td_operators."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import rlax

from rl_lib.td_operators import soft_watkins


class TestTDOperators(parameterized.TestCase):
    """Tests for TD operator functions."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(0)

        # T = 2 (number of transitions)
        self.T = 2
        self.num_actions = 2

        # q_s_full: Q-values at s_t for t=0,1,2. Shape (T+1, num_actions) = (3, 2)
        # Q(s_0), Q(s_1), Q(s_2)
        self.q_s_full = jnp.array(
            [
                [1.0, 1.2],
                [2.0, 2.2],
                [3.0, 3.2],
            ],
            dtype=jnp.float32,
        )

        # target_qs_full: Target Q-values at s_t for t=0,1,2. Shape (T+1, num_actions) = (3, 2)
        # TargetQ(s_0), TargetQ(s_1), TargetQ(s_2)
        self.target_qs_full = jnp.array(
            [
                [1.1, 1.3],
                [2.1, 2.3],
                [3.1, 3.3],
            ],
            dtype=jnp.float32,
        )

        # a_s: Actions taken at s_t for t=0,1 (a_0, a_1)
        # Shape (T,) = (2,)
        self.a_s_sequence = jnp.array([0, 1], dtype=jnp.int32)

        # r_s: Rewards r_{t+1} for t=0,1 (r_1, r_2)
        # Shape (T,) = (2,)
        self.r_s_sequence = jnp.array([0.5, 0.8], dtype=jnp.float32)

        # discount_s: Discounts d_{t+1} for t=0,1 (d_1, d_2)
        # Shape (T,) = (2,)
        self.discount_s_sequence = jnp.array([0.99, 0.99], dtype=jnp.float32)

        self.kappa = 1.0
        self._lambda_param = 0.7

        # Inputs for soft_watkins (T transitions):
        # q_s_input: Q(s_0), Q(s_1) -> self.q_s_full[:-1]
        self.q_s_input = self.q_s_full[:-1]  # Shape (T, num_actions)
        # target_qs_input: TargetQ(s_1), TargetQ(s_2) -> self.target_qs_full[1:]
        self.target_qs_input = self.target_qs_full[1:]  # Shape (T, num_actions)


    def test_soft_watkins_basic_case(self):
        """Tests basic invocation of soft_watkins for shape and type."""
        td_errors = soft_watkins(
            q_s=self.q_s_input,
            a_s=self.a_s_sequence,
            r_s=self.r_s_sequence,
            discount_s=self.discount_s_sequence,
            target_qs=self.target_qs_input,
            kappa=self.kappa,
            _lambda=self._lambda_param,
        )
        self.assertIsInstance(td_errors, jax.Array)
        # rlax.q_lambda returns TD error for each of the T transitions
        self.assertEqual(td_errors.shape, (self.T,))

    def test_soft_watkins_lambda_zero(self):
        """Tests soft_watkins with _lambda = 0, expecting 1-step TD errors."""
        _lambda_zero = 0.0

        td_errors_sw_lambda_zero = soft_watkins(
            q_s=self.q_s_input,
            a_s=self.a_s_sequence,
            r_s=self.r_s_sequence,
            discount_s=self.discount_s_sequence,
            target_qs=self.target_qs_input,
            kappa=self.kappa,
            _lambda=_lambda_zero,
        )

        # Calculate expected 1-step TD errors using rlax.q_learning
        # q_tm1 = self.q_s_input (Q-values at s_0, s_1)
        # a_t = self.a_s_sequence (actions at s_0, s_1)
        # r_t = self.r_s_sequence (rewards r_1, r_2)
        # discount_t = self.discount_s_sequence (discounts d_1, d_2)
        # q_t = self.target_qs_input (Target Q-values at s_1, s_2)

        # Vmap rlax.q_learning as it expects unbatched ranks for its core assertion.
        q_learning_fn_vmapped = jax.vmap(
            rlax.q_learning, in_axes=(0, 0, 0, 0, 0) # q_tm1, a_t, r_t, discount_t, q_t
        )
        expected_one_step_td_errors = q_learning_fn_vmapped(
            self.q_s_input,
            self.a_s_sequence,
            self.r_s_sequence,
            self.discount_s_sequence,
            self.target_qs_input,
        )

        self.assertIsInstance(td_errors_sw_lambda_zero, jax.Array)
        self.assertEqual(td_errors_sw_lambda_zero.shape, (self.T,))
        self.assertTrue(
            jnp.allclose(
                td_errors_sw_lambda_zero, expected_one_step_td_errors, atol=1e-6
            ),
            f"soft_watkins with lambda=0 ({td_errors_sw_lambda_zero}) "
            f"did not match expected 1-step TD errors ({expected_one_step_td_errors})",
        )


if __name__ == "__main__":
    absltest.main()