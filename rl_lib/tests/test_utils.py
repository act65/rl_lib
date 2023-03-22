from absl.testing import absltest
from absl.testing import parameterized

from rl_lib.utils import EMATree

import jax

import haiku as hk
import jax.numpy as jnp

class TestUtils(parameterized.TestCase):
    def test_ema(self, k=5):
        ema = EMATree(0.5)
        rng = hk.PRNGSequence(jax.random.PRNGKey(0))

        def get_val():
            return {'a': jnp.ones(()),
                    'b': jnp.ones(())}

        x = get_val()
        state = ema.init(x)

        for _ in range(k):
            x = get_val()
            state = ema(x, state)

        self.assertEqual(state['a'], 1.0)


if __name__ == '__main__':
    absltest.main()