import jax
import jax.numpy as jnp
import optax
import rlax

from jax.example_libraries import optimizers
from rl_lib.td_operators import soft_watkins
from rl_lib.utils import EMATree, cost, l2_regulariser

class RLAgent():
    """
    Supports;
    - target network
    - value learning
    """
    def __init__(self):
        raise NotImplementedError

    def initial_train_state(self, key):
        raise NotImplementedError

    def initial_actor_state(self):
        raise NotImplementedError

    def actor_step(self):
        raise NotImplementedError

    def learner_step(self, params, data, train_state, unused_key):
        raise NotImplementedError

class Random(RLAgent):
    """
    Implements a random policy
    """
    def __init__(self, action_spec):
        self._action_spec = action_spec

    def initial_train_state(self, key):
        return dict()

    def initial_actor_state(self):
        return ()

    def actor_step(self, params, observation, actor_state, key, evaluation):
        del params, observation, evaluation
        return jax.random.randint(key, (), 0, self._action_spec.num_values), actor_state

    def learner_step(self, train_state, data, key):
        del data, key
        return train_state, 0.0

class QLearner(RLAgent):
    """
    Implements;
    - Q-learning
    """
    def __init__(self, network, observation_spec, learning_rate):
        self._network = network
        self._batch_apply_net = jax.vmap(network.apply, in_axes=(None, 0))
        self._observation_spec = observation_spec
        self._learing_rate = learning_rate
        self._optimizer = optax.adam(learning_rate)
        
        # Jitting for speed.
        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)

    def initial_train_state(self, key):
        sample_input = self._observation_spec.generate_value()
        params = self._network.init(key, sample_input)
        opt_state = self._optimizer.init(params)
        return dict(
            params=params,
            opt_state=opt_state)
    
    def initial_actor_state(self):
        return ()

    def actor_step(self, train_state, obs, actor_state, key, evaluation):
        params = train_state['params']
        q_values = self._network.apply(params, obs)

        train_a = rlax.softmax().sample(key, q_values)
        eval_a = rlax.greedy().sample(key, q_values)

        return jax.lax.select(evaluation, eval_a, train_a), actor_state

    def _loss(self, params, states, actions, rewards, discounts, next_states, target_params=None):
        q_s = self._batch_apply_net(params, states)
        if target_params is None:
            target_qs = self._batch_apply_net(params, next_states)
        else:
            target_qs = self._batch_apply_net(target_params, next_states)
        td_error = self._td_operator(q_s, actions, rewards, discounts, target_qs)
        return cost(td_error) + l2_regulariser(params)
    
    def _td_operator(self, q_tm1, a_t, r_t, discount_t, q_t):
        q_learning = jax.vmap(rlax.q_learning, in_axes=(0, 0, 0, 0, 0))
        return q_learning(q_tm1, a_t, r_t, discount_t, q_t)

    def learner_step(self, state, batch, unused_key):
        grad_fn = jax.value_and_grad(self._loss)
        loss, grads = grad_fn(state['params'], **batch)
        grads = optimizers.clip_grads(grads, 1.0)
        updates, opt_state = self._optimizer.update(grads, state['opt_state'])
        new_params = optax.apply_updates(state['params'], updates)
        return dict(
            params=new_params, 
            opt_state=opt_state), loss

class QLambda(QLearner):
    """
    """
    def __init__(self, _lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lambda = _lambda  
   
    def _loss(self, params, states, actions, rewards, discounts, next_states, target_params=None):
        q_s = jax.lax.map(lambda x: self._batch_apply_net(params, x), states)
        if target_params is None:
            target_qs = jax.lax.map(lambda x: self._batch_apply_net(params, x), next_states)
        else:
            target_qs = jax.lax.map(lambda obs: self._batch_apply_net(target_params, obs), next_states)
        td_error = self._td_operator(q_s, actions, rewards, discounts, target_qs)
        return cost(td_error) + l2_regulariser(params)
    
    def _td_operator(self, q_s, a_s, r_s, discount_s, target_qs):
        _lambda = jnp.ones_like(discount_s)
        batch_soft_watkins = jax.vmap(rlax.q_lambda, in_axes=(0, 0, 0, 0, 0, 0))
        return batch_soft_watkins(q_s, a_s, r_s, discount_s, target_qs, _lambda)

class SoftWatkins(QLambda):
    """
    """
    def __init__(self, kappa, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = kappa

    def _td_operator(self, q_s, a_s, r_s, discount_s, target_qs):
        batch_soft_watkins = jax.vmap(soft_watkins, in_axes=(0, 0, 0, 0, 0, None, None))
        return batch_soft_watkins(q_s, a_s, r_s, discount_s, target_qs, self.kappa, self._lambda)
    
class EMATargetNet(SoftWatkins):
    def __init__(self, ema_decay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_fn = EMATree(ema_decay)

    def initial_train_state(self, key):
        state = super().initial_train_state(key)
        ema_state = self.ema_fn.init(state['params'])
        return dict(**state, ema_state=ema_state)

    def learner_step(self, state, batch, unused_key):
        ema = self.ema_fn(state['ema_state'], state['params'])
        batch.update({'target_params': ema})  # the 'batch' args are passed to _loss
        state, loss = super().learner_step(state, batch, unused_key)
        return dict(**state, ema_state=ema), loss

class MultiActionManager():
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def iterate(self, nn_output):
        raise NotImplementedError
    
    def get_mask(self, action):
        raise NotImplementedError

class MaskedMultiAction(EMATargetNet):
    """
    For the case where we have a single agent that is required to make multiple actions.
    Or there is some kind of structured, discrete, action space.
    And we want to share a core network between the actions.
    """
    def __init__(self, mam, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.mam = mam

    def actor_step(self, train_state, obs, actor_state, key, evaluation):
        params = train_state['params']
        nn_output = self._network.apply(params, obs)

        chosen_actions = []
        for q_values in self.mam.iterate(nn_output):      
            key, subkey = jax.random.split(key)
            train_a = rlax.softmax().sample(subkey, q_values)
            eval_a = rlax.greedy().sample(subkey, q_values)
            a = jax.lax.select(evaluation, eval_a, train_a)
            chosen_actions.append(a)

        return chosen_actions, actor_state

    def _loss(self, params, states, actions, rewards, discounts, next_states, target_params=None):
        q_s = jax.lax.map(lambda x: self._batch_apply_net(params, x), states)
        if target_params is None:
            target_qs = jax.lax.map(lambda x: self._batch_apply_net(params, x), next_states)
        else:
            target_qs = jax.lax.map(lambda obs: self._batch_apply_net(target_params, obs), next_states)

        loss = 0.0
        mask = jax.vmap(self.mam.get_mask)(actions)

        for i, (q, target_q) in enumerate(zip(self.mam.iterate(q_s), self.mam.iterate(target_qs))):
            td_error = self._td_operator(q, actions[..., i], rewards, discounts, target_q)
            loss += cost(td_error * mask[..., i])

        return loss + l2_regulariser(params)
