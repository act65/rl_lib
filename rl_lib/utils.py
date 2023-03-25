import tensorflow as tf
import jax.numpy as jnp
import flax.linen as nn
import jax

from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves
import socket

def build_signature(obs_spec, action_spec, n_multistep):
    """"
    Util for building the tf signature of the bsuite envs.
    """
    base_shape = tuple(s for s in (n_multistep, ))

    return {
        'states': tf.TensorSpec(shape=base_shape+obs_spec.shape, dtype=obs_spec.dtype),
        'actions': tf.TensorSpec(shape=base_shape+action_spec.shape, dtype=action_spec.dtype),
        'rewards': tf.TensorSpec(shape=base_shape+(), dtype=tf.float32),
        'discounts': tf.TensorSpec(shape=base_shape+(), dtype=tf.float32),
        'next_states': tf.TensorSpec(shape=base_shape+obs_spec.shape, dtype=obs_spec.dtype),
    }

class NN(nn.Module):
    n_dense: int
    width: int
    output_dims: int
    
    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, -1)
        for _ in range(self.n_dense-1):
            x = nn.Dense(
                features=self.width,
                kernel_init=jax.nn.initializers.orthogonal()
                )(x)
            x = nn.relu(x)
        qs = nn.Dense(features=self.output_dims)(x)
        return qs

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class EMA():
    def __init__(self, decay):
        self.decay = decay
 
    def __call__(self, state, value):
        decay = jax.lax.convert_element_type(self.decay, value.dtype)
        one = jnp.ones([], value.dtype)
        return state * decay + value * (one - decay)

class EMATree():
    def __init__(self, ema_decay):
        self.ema_fn = EMA(ema_decay)

    def init(self, tree):
        return tree
        
    def __call__(self, value, state):
        flat_state, _ = tree_flatten(state)
        flat_value, tree_struct = tree_flatten(value)

        # check trees are the same
        # assert tree_structure_a == tree_structure_b

        flat_avg = map(lambda x: self.ema_fn(x[0], x[1]), zip(flat_state, flat_value))
        return tree_unflatten(tree_struct, flat_avg)
    
def cost(error):
    """
    A cost function.
    Includes;
    - cubic errors cos they are supposed to be equivalent to proiority replay sampling (w mse of td error). ([ref](https://arxiv.org/abs/2007.09569))
    - abs errors cos they will ensure progress in v flat regions.
    - squared errors cos.
    """
    return (
        jnp.mean(jnp.abs(error))
        +jnp.mean(jnp.abs(error)**2) 
        +jnp.mean(jnp.abs(error)**3)
        )/3

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()

def l2_regulariser(params):
    return sum(
        l2_loss(w, alpha=0.001) 
        for w in tree_leaves(params)
    )