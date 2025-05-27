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

        # Common data for tests
        # T = 2 (sequence length for rlax.q_lambda which processes T transitions)
        # Thus, Q-value arrays (q_s, target_qs) often have T+1 elements
        self.T = 2
        self.num_actions = 2

        # q_s: Q-values at current steps s_t, s_{t+1}, s_{t+2}
        # Shape (T+1, num_actions) = (3, 2)
        self.q_s = jnp.array(
            [
                [1.0, 1.2],  # Q(s_t, a)
                [2.0, 2.2],  # Q(s_{t+1}, a)
                [3.0, 3.2],  # Q(s_{t+2}, a) - for bootstrapping last step of q_lambda
            ],
            dtype=jnp.float32,
        )

        # target_qs: Q-values at next steps, typically from a target network
        # Shape (T+1, num_actions) = (3, 2)
        self.target_qs = jnp.array(
            [
                [
                    1.1,
                    1.3,
                ],  # Target Q(s_t, a) - not directly used by q_lambda for this q_s
                [2.1, 2.3],  # Target Q(s_{t+1}, a)
                [3.1, 3.3],  # Target Q(s_{t+2}, a)
            ],
            dtype=jnp.float32,
        )

        # a_s: Actions taken at s_t, s_{t+1}
        # Shape (T,) = (2,)
        self.a_s = jnp.array([0, 1], dtype=jnp.int32)

        # r_s: Rewards received for (s_t, a_t) -> r_{t+1} and (s_{t+1}, a_{t+1}) -> r_{t+2}
        # Shape (T,) = (2,)
        self.r_s = jnp.array([0.5, 0.8], dtype=jnp.float32)

        # discount_s: Discounts for transitions starting at s_t, s_{t+1}
        # Shape (T,) = (2,)
        self.discount_s = jnp.array([0.99, 0.99], dtype=jnp.float32)

        self.kappa = 1.0
        self._lambda_param = 0.7

    def test_soft_watkins_basic_case(self):
        """Tests basic invocation of soft_watkins for shape and type."""
        td_errors = soft_watkins(
            q_s=self.q_s,
            a_s=self.a_s,
            r_s=self.r_s,
            discount_s=self.discount_s,
            target_qs=self.target_qs,
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
            q_s=self.q_s,
            a_s=self.a_s,
            r_s=self.r_s,
            discount_s=self.discount_s,
            target_qs=self.target_qs,
            kappa=self.kappa,
            _lambda=_lambda_zero,
        )

        # Calculate expected 1-step TD errors using rlax.q_learning for reference
        # q_s[:-1] are the Q-values for s_t, s_{t+1}
        # a_s are actions at s_t, s_{t+1}
        # r_s are rewards r_{t+1}, r_{t+2}
        # discount_s are discounts d_{t+1}, d_{t+2}
        # target_qs[1:] are the Q-values for s_{t+1}, s_{t+2} from the target network

        # Note: rlax.q_learning expects:
        # q_tm1: Q-values at state s_t. Shape: (B, A) or (A,).
        # a_t: Action taken at state s_t. Shape: (B,) or ().
        # r_t: Reward received after action a_t. Shape: (B,) or ().
        # discount_t: Discount received after action a_t. Shape: (B,) or ().
        # q_t: Q-values at state s_{t+1}. Shape: (B, A) or (A,).

        # For the first transition (t=0):
        # q_tm1 = self.q_s[0] -> Q(s_0, a)
        # a_t = self.a_s[0] -> a_0
        # r_t = self.r_s[0] -> r_1
        # discount_t = self.discount_s[0] -> d_1
        # q_t = self.target_qs[1] -> Target Q(s_1, a)

        # For the second transition (t=1):
        # q_tm1 = self.q_s[1] -> Q(s_1, a)
        # a_t = self.a_s[1] -> a_1
        # r_t = self.r_s[1] -> r_2
        # discount_t = self.discount_s[1] -> d_2
        # q_t = self.target_qs[2] -> Target Q(s_2, a)

        # So, we need to align inputs for a batched call to rlax.q_learning
        q_tm1_for_q_learning = self.q_s[:-1]  # Q(s_t) for t=0,1. Shape (2, 2)
        a_t_for_q_learning = self.a_s  # a_t for t=0,1. Shape (2,)
        r_t_for_q_learning = self.r_s  # r_{t+1} for t=0,1. Shape (2,)
        discount_t_for_q_learning = self.discount_s  # d_{t+1} for t=0,1. Shape (2,)
        q_t_for_q_learning = self.target_qs[
            1:
        ]  # Target Q(s_{t+1}) for t=0,1. Shape (2, 2)

        # rlax.q_learning returns (target - q_selected_action)
        # target = r_t + discount_t * max_a Q_target(s_{t+1}, a)
        expected_one_step_td_errors = rlax.q_learning(
            q_tm1_for_q_learning,
            a_t_for_q_learning,
            r_t_for_q_learning,
            discount_t_for_q_learning,
            q_t_for_q_learning,
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
