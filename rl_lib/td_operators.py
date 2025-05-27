"""This module implements various Temporal Difference (TD) learning operators.

These operators are fundamental components in many reinforcement learning
algorithms, used to calculate TD errors or construct TD targets.
"""

import jax
import jax.numpy as jnp
import rlax
from typing import Callable  # Added Callable for vectorized_lambda_at_t

# Renamed jnp.ndarray to jax.Array for consistency with JAX's own type hints
# This is a common practice seen in JAX documentation and examples.
# If jnp.ndarray is preferred, it can be used directly.
Array = jax.Array  # Using jax.Array as a more general JAX array type


def bellman_uncertainty() -> None:
    """Placeholder for Bellman uncertainty calculations.

    This function is intended to implement mechanisms for incorporating
    uncertainty into Bellman updates, which can be useful for exploration
    or robust decision-making. It is not yet implemented.

    Raises:
        NotImplementedError: Always, as the feature is not yet implemented.
    """
    raise NotImplementedError(
        "bellman_uncertainty is a planned feature and is not yet implemented."
    )


def soft_watkins(
    q_s: Array,
    a_s: Array,
    r_s: Array,
    discount_s: Array,
    target_qs: Array,
    kappa: float,
    _lambda: float,
) -> Array:
    """Implements the Soft Watkins Q(lambda) operator.

    This function calculates the TD error for a sequence of transitions using
    the Q(lambda) algorithm with a "soft" Watkins-style off-policy correction.
    The lambda value for each timestep is dynamically calculated based on the
    current policy's Q-values and a kappa parameter, which determines how
    closely the policy needs to match the target policy for lambda to be non-zero.

    The calculation of `lambda_t` is based on the paper:
    "Safe and Efficient Off-Policy Reinforcement Learning" (Munos et al., 2016)
    (often referenced in discussions of Retrace(lambda) and related algorithms).
    Specifically, it's related to equation:
    λₜ = λ * E_{a ~ π(.|x_t)} [ I{Q(x_t, a_t) >= Q(x_t, a) - κ * |Q(x_t, a)|} ]
    as described in https://openreview.net/pdf?id=JtC6yOHRoJJ (Figure 1, though
    this code uses Q(x_t, a_t) instead of Q(x_t, a_max) as the reference).

    Args:
        q_s: A JAX array of Q-values at each state s_t in a sequence (trajectory).
            Assumed shape: (T, num_actions), where T is sequence length.
            These are the Q-values for the policy being evaluated.
        a_s: A JAX array of actions taken at each state s_t in the sequence.
            Assumed shape: (T,). These are the actions from the behavior policy.
            Note: rlax.q_lambda expects `a_tm1`, so this usually corresponds to
            actions from t=0 to T-1 if q_s is from t=0 to T-1.
        r_s: A JAX array of rewards received after taking action a_s at state s_t.
            Assumed shape: (T,).
            Note: rlax.q_lambda expects `r_t`, so this usually corresponds to
            rewards from t=1 to T if q_s is from t=0 to T-1.
        discount_s: A JAX array of discount factors for each transition.
            Assumed shape: (T,).
        target_qs: A JAX array of Q-values at the next state s_{t+1},
            typically from a target network.
            Assumed shape: (T, num_actions).
        kappa: A float parameter controlling the "softness" of the Watkins
            condition. A smaller kappa makes the condition stricter (lambda_t is
            more likely to be zero if the greedy action under `target_qs`
            differs from `a_s`).
        _lambda: The base lambda value (float) for the Q(lambda) algorithm.
            This is scaled by the per-timestep calculated factor.

    Returns:
        A JAX array representing the TD errors for the sequence.
        The shape is typically (T-1,) or (T,) depending on rlax.q_lambda details.
    """

    def _calculate_lambda_t_values(
        q_target_at_step_t: Array,
        action_at_step_t: Array,  # Changed from int to Array (scalar JAX array)
        kappa_val: float,
        base_lambda_val: float,
    ) -> Array:  # Return Array (scalar JAX array)
        """Calculates the per-timestep lambda value (lambda_t).

        This implements the soft Watkins condition:
        λₜ = base_λ * E_{a ~ π(.|x_t)} [ I{Q(x_t, a_t) >= Q(x_t, a) - κ * |Q(x_t, a)|} ]
        where Q refers to `q_target_at_step_t`, a_t is `action_at_step_t`,
        and π is the softmax policy derived from `q_target_at_step_t`.

        Args:
            q_target_at_step_t: Target Q-values for all actions at a single timestep. Shape (num_actions,).
            action_at_step_t: The action taken at that timestep. Scalar JAX array.
            kappa_val: The kappa parameter for softness.
            base_lambda_val: The base lambda value.

        Returns:
            The calculated lambda_t for that timestep (scalar JAX array).
        """
        # Indicator function: I{Q(s_t, a_t) >= Q(s_t, a) - κ * |Q(s_t, a)|} for all 'a'
        # q_target_at_step_t[action_at_step_t] is Q(s_t, a_t)
        # q_target_at_step_t is Q(s_t, a) for all 'a'
        indicator_fn_values: Array = jnp.where(
            q_target_at_step_t[action_at_step_t]
            >= q_target_at_step_t - kappa_val * jnp.abs(q_target_at_step_t),
            1.0,
            0.0,
        )
        # Policy π(a|s_t) derived from target Q-values (e.g., softmax)
        pi_probs: Array = rlax.softmax().probs(q_target_at_step_t)

        # Expected value of the indicator function under policy π
        expected_indicator: Array = jnp.inner(pi_probs, indicator_fn_values)

        lambda_t_value: Array = base_lambda_val * expected_indicator
        return lambda_t_value

    # Vectorize the calculation of lambda_t over the time dimension (sequence length)
    # We apply this to target_qs and target_as (actions derived from target_qs)
    # The vectorized function will take (T, num_actions), (T,), None, None and return (T,)
    vectorized_lambda_at_t: Callable[[Array, Array, float, float], Array] = jax.vmap(
        _calculate_lambda_t_values, in_axes=(0, 0, None, None)
    )

    # Determine target actions for lambda calculation.
    # The action_at_step_t for _calculate_lambda_t_values should be the action
    # whose Q-value Q(s_t, action_at_step_t) serves as the reference in the indicator.
    # This is typically the greedy action under the target Q-values (target_qs).
    greedy_target_actions: Array = jnp.argmax(target_qs, axis=-1)  # Shape (T,)

    # Calculate lambda_t for each step in the sequence
    # lambda_values will have shape (T,)
    lambda_values: Array = vectorized_lambda_at_t(
        target_qs, greedy_target_actions, kappa, _lambda
    )

    # rlax.q_lambda computes Peng's Q(lambda) errors.
    # q_tm1: Q-values at state s_t. (T, num_actions)
    # a_tm1: Actions taken at state s_t. (T,)
    # r_t: Rewards received after taking a_tm1, leading to s_{t+1}. (T,)
    # discount_t: Discount factor for transition s_t -> s_{t+1}. (T,)
    # q_t: Q-values at state s_{t+1} (target Q-values). (T, num_actions)
    # lambda_: The mixing parameter, can be scalar or array (T,).
    return rlax.q_lambda(
        q_tm1=q_s,
        a_tm1=a_s,  # Actions from behavior policy
        r_t=r_s,
        discount_t=discount_s,
        q_t=target_qs,  # Target Q-values for next states
        lambda_=lambda_values,  # Our dynamically calculated lambda values
    )
