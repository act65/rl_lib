"""Utility functions and classes supporting the Reinforcement Learning library.

This module provides a collection of helper tools, including:
- Functions for building Reverb table signatures (`build_signature`).
- Network utilities like a simple FeedForward Neural Network (`NN`).
- Port checking utilities (`is_port_in_use`).
- Exponential Moving Average (EMA) calculation classes (`EMA`, `EMATree`).
- Custom cost and regularization functions (`cost`, `l2_loss`, `l2_regulariser`).
"""

import tensorflow as tf
import jax.numpy as jnp
import flax.linen as nn
import jax

# Import jax.Array directly for type hinting JAX arrays
from jax import Array as JaxArray

# For PyTree type hint, try jax.typing.PyTree, fallback to Any
try:
    from jax.typing import PyTree
except ImportError:
    from typing import Any as PyTree  # type: ignore

from typing import Any, Dict, List, Tuple  # Added Iterable

from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, PyTreeDef
import socket

# Define a more specific type for obs_spec and action_spec if possible,
# otherwise Any or a custom Protocol can be used.
# For now, using Any as their structure is not fully defined here.
Spec = Any


def build_signature(
    obs_spec: Spec, action_spec: Spec, n_multistep: int
) -> Dict[str, tf.TensorSpec]:
    """Constructs a Reverb table signature for N-step transitions.

    This utility function creates a dictionary of `tf.TensorSpec` objects
    that defines the structure of data to be stored in a Reverb table.
    The signature includes states, actions, rewards, discounts, and next_states,
    all prefixed with an N-step sequence dimension.

    Args:
        obs_spec: A `tf.TensorSpec` describing a single observation from the
            environment. Its shape and dtype are used for 'states' and 'next_states'.
        action_spec: A `tf.TensorSpec` describing a single action from the
            environment. Its shape and dtype are used for 'actions'.
        n_multistep: An integer representing the number of steps in the
            transition sequence (N). This will be the leading dimension for all
            elements in the signature.

    Returns:
        A dictionary where keys are strings ('states', 'actions', 'rewards',
        'discounts', 'next_states') and values are `tf.TensorSpec` instances.
        Each TensorSpec will have `(n_multistep,)` as its leading dimensions,
        followed by the original shape of `obs_spec` or `action_spec`.
        Rewards and discounts are scalar per step, so their shape will be
        `(n_multistep,)`.
    """
    base_shape: Tuple[int, ...] = (n_multistep,)

    return {
        "states": tf.TensorSpec(
            shape=base_shape + obs_spec.shape, dtype=obs_spec.dtype
        ),
        "actions": tf.TensorSpec(
            shape=base_shape + action_spec.shape, dtype=action_spec.dtype
        ),
        "rewards": tf.TensorSpec(shape=base_shape, dtype=tf.float32),
        "discounts": tf.TensorSpec(shape=base_shape, dtype=tf.float32),
        "next_states": tf.TensorSpec(
            shape=base_shape + obs_spec.shape, dtype=obs_spec.dtype
        ),
    }


class NN(nn.Module):
    """A simple FeedForward Neural Network using Flax.

    This network consists of a configurable number of dense layers with ReLU
    activations, followed by a final dense output layer without activation.
    The input is first flattened. Kernels are initialized orthogonally.

    Attributes:
        n_dense (int): The total number of dense layers, including the output layer.
                       Must be at least 1.
        width (int): The number of units in each hidden dense layer.
        output_dims (int): The number of output units (dimensionality of the output).
    """

    n_dense: int
    width: int
    output_dims: int

    @nn.compact
    def __call__(self, x: JaxArray) -> JaxArray:
        """Forward pass of the neural network.

        Args:
            x: The input JAX array. It will be reshaped to (-1,) before processing.
               Shape: Can be any shape that is compatible with `jnp.reshape(x, -1)`.

        Returns:
            A JAX array representing the output of the network (Q-values or other predictions).
            Shape: (output_dims,).
        """
        if self.n_dense < 1:
            raise ValueError("n_dense must be at least 1 (for the output layer).")

        processed_x: JaxArray = jnp.reshape(x, -1)

        for _ in range(self.n_dense - 1):
            processed_x = nn.Dense(
                features=self.width, kernel_init=jax.nn.initializers.orthogonal()
            )(processed_x)
            processed_x = nn.relu(processed_x)

        output_qs: JaxArray = nn.Dense(
            features=self.output_dims, kernel_init=jax.nn.initializers.orthogonal()
        )(processed_x)
        return output_qs


def is_port_in_use(port: int) -> bool:
    """Checks if a given TCP port is currently in use on localhost.

    Args:
        port: The port number to check.

    Returns:
        True if the port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


class EMA:
    """Calculates an Exponential Moving Average (EMA) for a single JAX array.

    This class provides a callable interface to update an EMA state with a new value.
    The decay rate is specified at initialization.
    """

    decay: float

    def __init__(self, decay: float) -> None:
        """Initializes the EMA calculator.

        Args:
            decay: The decay rate for the EMA. A higher value means slower decay,
                   giving more weight to past states. Typically between 0 and 1.
        """
        self.decay = decay

    def __call__(self, state: JaxArray, value: JaxArray) -> JaxArray:
        """Updates the EMA with a new value.

        The formula used is: `new_state = state * decay + value * (1 - decay)`.
        The decay rate is converted to the dtype of the `value`.

        Args:
            state: The current EMA state (the moving average value).
                   Shape must match `value`.
            value: The new value to incorporate into the EMA.
                   Shape must match `state`.

        Returns:
            The updated EMA state. Shape and dtype match `value`.
        """
        decay_typed: JaxArray = jax.lax.convert_element_type(self.decay, value.dtype)
        one: JaxArray = jnp.ones([], value.dtype)
        return state * decay_typed + value * (one - decay_typed)


class EMATree:
    """Applies Exponential Moving Average (EMA) to all leaves of a PyTree.

    This class facilitates maintaining an EMA for complex data structures like
    neural network parameters (which are often PyTrees). It uses the `EMA`
    class internally for each leaf.
    """

    ema_fn: EMA

    def __init__(self, ema_decay: float) -> None:
        """Initializes the EMATree calculator.

        Args:
            ema_decay: The decay rate for the EMA, passed to the internal
                       `EMA` instance for each leaf.
        """
        self.ema_fn = EMA(ema_decay)

    def init(self, tree: PyTree) -> PyTree:
        """Initializes the EMA state for a PyTree.

        For EMA, the initial state is typically the tree itself.

        Args:
            tree: The PyTree for which to initialize the EMA state.

        Returns:
            The initial EMA state, which is identical to the input `tree`.
        """
        return tree

    def __call__(self, state_tree: PyTree, value_tree: PyTree) -> PyTree:
        """Updates the EMA for each leaf in a PyTree.

        It flattens both the current EMA state tree and the new value tree,
        applies the EMA update to corresponding leaves, and then reconstructs
        the tree with the updated EMA values.

        Args:
            state_tree: The PyTree representing the current EMA states for each leaf.
            value_tree: The PyTree of new values to incorporate into the EMA.
                        Must have the same tree structure as `state_tree`.

        Returns:
            A new PyTree with the updated EMA values for each leaf.
            The tree structure matches `value_tree`.

        Raises:
            TypeError: If the tree structures of `state_tree` and `value_tree`
                       do not match (implicit via `jax.tree_util.tree_unflatten`).
        """
        flat_state_leaves: List[Any]
        state_treedef: PyTreeDef
        flat_state_leaves, state_treedef = tree_flatten(state_tree)

        flat_value_leaves: List[Any]
        value_treedef: PyTreeDef
        flat_value_leaves, value_treedef = tree_flatten(value_tree)

        if state_treedef != value_treedef:
            raise TypeError(
                "state_tree and value_tree must have the same Pytree structure."
            )

        updated_flat_leaves: List[JaxArray] = [
            self.ema_fn(s_leaf, v_leaf)
            for s_leaf, v_leaf in zip(flat_state_leaves, flat_value_leaves)
        ]

        return tree_unflatten(value_treedef, updated_flat_leaves)


def cost(error: JaxArray) -> JaxArray:
    """Computes a composite cost from prediction errors.

    The cost function is an average of three components:
    1.  Mean Absolute Error: `mean(|error|)`
        - Ensures progress in regions with very flat gradients.
    2.  Mean Squared Error: `mean(error^2)`
        - Standard component, penalizes larger errors more.
    3.  Mean Cubic Error: `mean(|error|^3)`
        - Further penalizes large errors. This component, when used with TD errors,
          is suggested to be related to prioritized experience replay when the
          priority is proportional to the TD error magnitude.
          (See "Recurrent Experience Replay in Distributed Reinforcement Learning" - R2D2, Appendix B.1,
           https://openreview.net/forum?id=r1lyTjAqYX, which references similar ideas from
           "Prioritized Experience Replay" https://arxiv.org/abs/1511.05952).
          The original code comment referenced https://arxiv.org/abs/2007.09569,
          which discusses prioritized sampling in general.

    Args:
        error: A JAX array of errors (e.g., TD errors). Can be of any shape
               for which `jnp.mean` is applicable.

    Returns:
        A scalar JAX array representing the computed cost.
    """
    mean_abs_error: JaxArray = jnp.mean(jnp.abs(error))
    mean_sq_error: JaxArray = jnp.mean(jnp.square(error))
    mean_cubic_error: JaxArray = jnp.mean(jnp.abs(error) ** 3)

    return (mean_abs_error + mean_sq_error + mean_cubic_error) / 3.0


def l2_loss(x: JaxArray, alpha: float) -> JaxArray:
    """Computes the L2 loss (mean of squares) scaled by a factor.

    Formula: `alpha * mean(x^2)`.

    Args:
        x: A JAX array for which to compute the L2 loss.
        alpha: A float scaling factor for the loss.

    Returns:
        A scalar JAX array representing the scaled L2 loss.
    """
    return alpha * jnp.mean(jnp.square(x))


def l2_regulariser(params: PyTree, alpha: float = 0.001) -> JaxArray:
    """Computes an L2 regularization term for a PyTree of parameters.

    It calculates the `l2_loss` for each leaf (e.g., weight matrix or bias vector)
    in the `params` PyTree and sums them up. This is commonly used to penalize
    large parameter values to prevent overfitting.

    Args:
        params: A PyTree (e.g., a Flax model's parameters) containing JAX arrays as leaves.
        alpha: The scaling factor for the L2 loss, applied to each leaf.
               Defaults to 0.001.

    Returns:
        A scalar JAX array representing the total L2 regularization term.
    """
    leaves: List[Any] = tree_leaves(params)
    # Ensure sum starts with a JAX array type for consistent computation if leaves is empty
    # or all leaves are None (though the filter below handles None).
    # However, sum() over an empty sequence of numbers is 0, which is fine.
    # If leaves can contain non-JaxArray types that l2_loss can't handle,
    # further filtering or type checking might be needed.
    sum_of_l2_losses: JaxArray = sum(  # type: ignore[assignment]
        l2_loss(w, alpha=alpha) for w in leaves if w is not None
    )
    return sum_of_l2_losses
