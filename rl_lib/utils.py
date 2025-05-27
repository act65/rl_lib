"""Utility functions and classes supporting the Reinforcement Learning library.
(Docstring unchanged)
"""

import tensorflow as tf
import jax.numpy as jnp
import flax.linen as nn
import jax

from jax import Array as JaxArray
try:
    from jax.typing import PyTree
except ImportError:
    from typing import Any as PyTree # type: ignore

from typing import Any, Dict, List, Tuple
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, PyTreeDef, tree_map # Import tree_map from tree_util
import socket

Spec = Any


def build_signature(
    obs_spec: Spec, action_spec: Spec, n_multistep: int
) -> Dict[str, tf.TensorSpec]:
    """Constructs a Reverb table signature for N-step transitions.
    (Docstring unchanged)
    """
    base_shape: Tuple[int, ...] = (n_multistep,)
    obs_shape = getattr(obs_spec, 'shape', ())
    obs_dtype = getattr(obs_spec, 'dtype', tf.float32)
    action_shape = getattr(action_spec, 'shape', ())
    action_dtype = getattr(action_spec, 'dtype', tf.int32)

    return {
        "states": tf.TensorSpec(
            shape=base_shape + obs_shape, dtype=obs_dtype
        ),
        "actions": tf.TensorSpec(
            shape=base_shape + action_shape, dtype=action_dtype
        ),
        "rewards": tf.TensorSpec(shape=base_shape, dtype=tf.float32),
        "discounts": tf.TensorSpec(shape=base_shape, dtype=tf.float32),
        "next_states": tf.TensorSpec(
            shape=base_shape + obs_shape, dtype=obs_dtype
        ),
    }


class NN(nn.Module):
    """A simple FeedForward Neural Network using Flax.
    (Docstring unchanged)
    """
    n_dense: int
    width: int
    output_dims: int

    @nn.compact
    def __call__(self, x: JaxArray) -> JaxArray:
        """Forward pass of the neural network.
        (Docstring unchanged)
        """
        if self.n_dense < 1:
            raise ValueError("n_dense must be at least 1 (for the output layer).")

        processed_x: JaxArray = jnp.asarray(x)
        processed_x = jnp.reshape(processed_x, -1)

        for i in range(self.n_dense - 1):
            processed_x = nn.Dense(
                features=self.width,
                kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                name=f'hidden_{i}'
            )(processed_x)
            processed_x = nn.relu(processed_x)

        output_qs: JaxArray = nn.Dense(
            features=self.output_dims,
            kernel_init=jax.nn.initializers.orthogonal(1.0),
            name='output_layer'
        )(processed_x)
        return output_qs


def is_port_in_use(port: int) -> bool:
    """Checks if a given TCP port is currently in use on localhost.
    (Docstring unchanged)
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(0.1)
            return s.connect_ex(("localhost", port)) == 0
        except (socket.timeout, ConnectionRefusedError):
            return False


class EMA:
    """Calculates an Exponential Moving Average (EMA) for a single JAX array.
    (Docstring unchanged)
    """
    decay: float

    def __init__(self, decay: float) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be between 0 and 1.")
        self.decay = decay

    def __call__(self, state: JaxArray, value: JaxArray) -> JaxArray:
        """Updates the EMA with a new value.
        (Docstring unchanged)
        """
        if not isinstance(state, JaxArray) or not isinstance(value, JaxArray):
            state = jnp.asarray(state)
            value = jnp.asarray(value)

        decay_typed: JaxArray = jnp.asarray(self.decay, dtype=value.dtype)
        one: JaxArray = jnp.ones_like(decay_typed)
        
        return state * decay_typed + value * (one - decay_typed)


class EMATree:
    """Applies Exponential Moving Average (EMA) to all leaves of a PyTree.
    (Docstring unchanged)
    """
    ema_fn: EMA

    def __init__(self, ema_decay: float) -> None:
        self.ema_fn = EMA(ema_decay)

    def init(self, tree: PyTree) -> PyTree:
        """Initializes the EMA state for a PyTree.
        (Docstring unchanged)
        """
        # FIX: Use jax.tree_util.tree_map
        return tree_map(jnp.asarray, tree)


    def __call__(self, state_tree: PyTree, value_tree: PyTree) -> PyTree:
        """Updates the EMA for each leaf in a PyTree.
        (Docstring unchanged, added structure check)
        """
        flat_state_leaves, state_treedef = tree_flatten(state_tree)
        flat_value_leaves, value_treedef = tree_flatten(value_tree)

        if state_treedef != value_treedef:
            raise TypeError(
                "state_tree and value_tree must have the same Pytree structure. "
                f"State: {state_treedef}, Value: {value_treedef}"
            )
        if not flat_value_leaves:
            return tree_unflatten(value_treedef, [])

        updated_flat_leaves: List[JaxArray] = [
            self.ema_fn(s_leaf, v_leaf)
            for s_leaf, v_leaf in zip(flat_state_leaves, flat_value_leaves)
        ]
        return tree_unflatten(value_treedef, updated_flat_leaves)


def cost(error: JaxArray) -> JaxArray:
    """Computes a composite cost from prediction errors.
    (Docstring unchanged)
    """
    error = jnp.asarray(error)
    mean_abs_error: JaxArray = jnp.mean(jnp.abs(error))
    mean_sq_error: JaxArray = jnp.mean(jnp.square(error))
    mean_cubic_error: JaxArray = jnp.mean(jnp.power(jnp.abs(error), 3))

    return (mean_abs_error + mean_sq_error + mean_cubic_error) / 3.0


def l2_loss(x: JaxArray, alpha: float) -> JaxArray:
    """Computes the L2 loss (mean of squares) scaled by a factor.
    (Docstring unchanged)
    """
    x = jnp.asarray(x)
    return alpha * jnp.mean(jnp.square(x))


def l2_regulariser(params: PyTree, alpha: float = 0.001) -> JaxArray:
    """Computes an L2 regularization term for a PyTree of parameters.
    (Docstring unchanged)
    """
    leaves: List[Any] = tree_leaves(params)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)

    # FIX: Replace jnp.addN with sum()
    # Ensure that we only sum actual JAX arrays that have a size.
    loss_terms = [l2_loss(w, alpha=alpha) for w in leaves if isinstance(w, JaxArray) and w.size > 0]
    if not loss_terms: # If no valid leaves found
        return jnp.array(0.0, dtype=jnp.float32)
    
    l2_sum = sum(loss_terms)
    return l2_sum