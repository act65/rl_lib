import unittest

import json
import numpy as np
import reverb

from rl_lib.utils import build_signature
from rl_lib.accumulators import ReverbAccumulator, MultiAgentReverbAccumulator

from absl.testing import absltest
from absl.testing import parameterized

from bsuite.environments import catch

from rl_lib.experiment import run_loop
from rl_lib.learners import Random

class TestReverb(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.env = catch.Catch(seed=0)

    @parameterized.parameters(
        {'batch_size': 1, 'n_multistep': 1},
        {'batch_size': 4, 'n_multistep': 1},
        {'batch_size': 1, 'n_multistep': 4},
        {'batch_size': 4, 'n_multistep': 4},
      )
    def test_reverb_sample(self, batch_size, n_multistep):

        signature = build_signature(
            self.env.observation_spec(),
            self.env.action_spec(),
            n_multistep
        )

        accumulator = ReverbAccumulator(
            port=5000, 
            signature=signature,
            max_size=10,
            n_multistep=n_multistep, 
            batch_size=batch_size, 
            shuffle_buffer_size=10)
        
        run_loop(
            agent=Random(self.env.action_spec()),
            environment=self.env,
            accumulator=accumulator,
            seed=0,
            batch_size=batch_size,
            train_episodes=10,
            evaluate_every=10,
            eval_episodes=1,
        )

        batch = accumulator.sample(batch_size)
        for k, v in batch.items():
            self.assertTrue(v.shape[0] == batch_size)
            self.assertTrue(v.shape[1] == n_multistep)

    @parameterized.parameters(
            {'n_agents': 2, 'n_multistep': 4, 'batch_size': 1},
            {'n_agents': 5, 'n_multistep': 4, 'batch_size': 1},
        )
    def test_multiagent(self, n_agents, n_multistep, batch_size):

        signature = build_signature(
                    self.env.observation_spec(),
                    self.env.action_spec(),
                    n_multistep
                )
                
        accumulator = MultiAgentReverbAccumulator(
                    port=5000, 
                    signature=signature,
                    max_size=10,
                    n_multistep=n_multistep, 
                    batch_size=batch_size, 
                    shuffle_buffer_size=10)

        obs = {str(i): self.env.observation_spec().generate_value() for i in range(n_agents)}
        acts = {str(i): self.env.action_spec().generate_value() for i in range(n_agents)}
        rews = {str(i): 0.0 for i in range(n_agents)}
        discounts = {str(i): 1.0 for i in range(n_agents)}

        # iterate n times so we add 1 example to the buffer
        for _ in range(n_multistep + 2):
            accumulator.push(obs, acts, rews, discounts)

        self.assertEqual(accumulator.current_size, n_agents)

if __name__ == '__main__':
    absltest.main()