"""This module implements various Temporal Difference (TD) learning operators.

These operators are fundamental components in many reinforcement learning
algorithms, used to calculate TD errors or construct TD targets.
"""

import jax
import jax.numpy as jnp
import rlax
from typing import Callable

Array = jax.Array


def bellman_uncertainty() -> None:
    """Placeholder for Bellman uncertainty calculations.
    (Docstring unchanged)
    """
    raise NotImplementedError(
        "bellman_uncertainty is a planned feature and is not yet implemented."
    )


def soft_watkins(
    q_s: Array,       # (seq_len, num_actions) - Q-values from behavior policy at s_t
    a_s: Array,       # (seq_len,) - actions taken by behavior policy at s_t
    r_s: Array,       # (seq_len,) - rewards received at s_{t+1} after (s_t, a_s)
    discount_s: Array,# (seq_len,) - discount at s_{t+1}
    target_qs: Array, # (seq_len, num_actions) - Target Q-values at s_{t+1}
    kappa: float,
    _lambda: float,
) -> Array:
    """Implements the Soft Watkins Q(lambda) operator.
    (Docstring largely unchanged, minor clarifications on arg meanings)

    Args:
        q_s: Q-values for the policy being evaluated at each state s_t. Shape (T, num_actions).
        a_s: Actions taken by the behavior policy at each state s_t. Shape (T,).
        r_s: Rewards received after (s_t, a_s), leading to s_{t+1}. Shape (T,).
        discount_s: Discount factors for transition s_t -> s_{t+1}. Shape (T,).
        target_qs: Target Q-values at state s_{t+1}. Shape (T, num_actions).
        kappa: Softness parameter for Watkins condition.
        _lambda: Base lambda value.

    Returns:
        TD errors for the sequence. Shape typically (T-1,) or (T,).
    """

    def _calculate_lambda_t_values(
        q_values_at_step_t: Array, # Q-values (e.g. target_qs) at a single step s_t. (num_actions,)
        action_reference_at_step_t: Array, # Reference action at s_t (e.g. greedy in target_qs). Scalar.
        kappa_val: float,
        base_lambda_val: float,
    ) -> Array: # Returns scalar lambda_t
        """Calculates the per-timestep lambda value (lambda_t)."""
        # Q(s_t, a_ref)
        q_value_of_reference_action = q_values_at_step_t[action_reference_at_step_t]

        # Indicator: I{Q(s_t, a_ref) >= Q(s_t, a) - κ * |Q(s_t, a)|} for all 'a'
        indicator_fn_values: Array = jnp.where(
            q_value_of_reference_action >= q_values_at_step_t - kappa_val * jnp.abs(q_values_at_step_t),
            1.0,
            0.0,
        )
        # Policy π(a|s_t) derived from q_values_at_step_t (e.g., softmax over target Qs)
        pi_probs: Array = rlax.softmax().probs(q_values_at_step_t)

        expected_indicator: Array = jnp.inner(pi_probs, indicator_fn_values)
        lambda_t_value: Array = base_lambda_val * expected_indicator
        return lambda_t_value

    # Vectorize over the time dimension (sequence length T)
    # The inputs to _calculate_lambda_t_values are per-timestep.
    # target_qs are Q(s_{t+1}), but the lambda_t calculation is for adjusting lambda at time t,
    # based on the policy derived from Q-values at s_t.
    # The original code used target_qs (Q values at s_{t+1}) and greedy_target_actions (actions at s_{t+1})
    # to calculate lambda_values that are then passed to rlax.q_lambda.
    # rlax.q_lambda's lambda_ parameter is "A scalar or a tensor of shape [T]".
    # If lambda_ is a tensor, lambda_[i] is the amount to bootstrap for the i-th transition.
    # So, lambda_values[t] should be based on information at time t (or t+1 depending on convention).
    # The original code calculates lambda_values based on target_qs (Q(s_{t+1})).
    # Let's stick to the original logic: lambda_t is based on Q-values at s_{t+1} (target_qs).
    # This means lambda_values[i] corresponds to the transition from s_i to s_{i+1}.

    # greedy_actions_from_target_qs are actions that would be taken at s_{t+1} if greedy w.r.t target_qs.
    # Shape (T,)
    greedy_actions_from_target_qs: Array = jnp.argmax(target_qs, axis=-1)

    # Calculate lambda_t for each step in the sequence.
    # lambda_values will have shape (T,). lambda_values[i] is for transition i.
    vectorized_lambda_calculator: Callable[[Array, Array, float, float], Array] = jax.vmap(
        _calculate_lambda_t_values, in_axes=(0, 0, None, None) # (target_qs_step, action_ref_step, k, l)
    )
    lambda_values: Array = vectorized_lambda_calculator(
        target_qs, # Q-values at s_{t+1}
        greedy_actions_from_target_qs, # Greedy actions at s_{t+1} based on target_qs
        kappa,
        _lambda
    )

    # rlax.q_lambda arguments:
    # q_tm1: Q-values at state s_t. (T, num_actions) -> q_s
    # a_tm1: Actions taken at state s_t. (T,) -> a_s
    # r_t: Rewards received after taking a_tm1, at s_{t+1}. (T,) -> r_s
    # discount_t: Discount factor for transition s_t -> s_{t+1}. (T,) -> discount_s
    # q_t: Q-values at state s_{t+1} (target Q-values). (T, num_actions) -> target_qs
    # lambda_: The mixing parameter, can be scalar or array (T,). -> lambda_values
    return rlax.q_lambda(
        q_tm1=q_s,
        a_tm1=a_s,
        r_t=r_s,
        discount_t=discount_s,
        q_t=target_qs,
        lambda_=lambda_values,
    )