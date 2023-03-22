import tensorflow as tf
import jax.numpy as jnp
import flax.linen as nn
import jax

import socket

def build_signature(obs_spec, action_spec, n_multistep):
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
        for _ in range(self.n_dense):
            x = nn.Dense(
                features=self.width,
                kernel_init=jax.nn.initializers.orthogonal()
                )(x)
            x = nn.selu(x)
        qs = nn.Dense(features=self.output_dims)(x)
        return qs

# def build_network(num_hidden_units: int, num_actions: int) -> hk.Transformed:
#     """Factory for a network for Q-values."""
#     def q(obs):
#         flatten = lambda x: jnp.reshape(x, (-1,))
#         network = hk.Sequential(
#             [flatten, nets.MLP([num_hidden_units, num_actions])])
    
#         return network(obs)

#     return hk.without_apply_rng(hk.transform(q))

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
