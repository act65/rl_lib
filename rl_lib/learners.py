"""This module defines various Reinforcement Learning (RL) agent architectures.

It includes base classes like RLAgent, and specific implementations such as
Random agents, Q-learning based agents (QLearner, QLambda, SoftWatkins),
and agents with features like EMA target networks (EMATargetNet) and
support for multi-action environments (MaskedMultiAction).
The agents are designed to be compatible with JAX and Optax for optimization.
"""

import jax
import jax.numpy as jnp
import optax
import rlax
import flax.linen as nn # Replaced haiku as hk

# from jax.example_libraries import optimizers # Replaced with optax clipping
from rl_lib.td_operators import soft_watkins
from rl_lib.utils import EMATree, cost, l2_regulariser

from typing import Any, Dict, Tuple, Callable, Optional, Iterator, List

# For PyTree type hint, try jax.typing.PyTree, fallback to Any
try:
    from jax.typing import PyTree
except ImportError:
    from typing import Any as PyTree # type: ignore


# Define common type aliases for clarity
PRNGKey = jax.Array
Params = PyTree # Replaced hk.Params
OptState = optax.OptState
Observation = jax.Array
Action = jax.Array # Can be int for discrete, or Array for continuous/multi
Loss = jax.Array
ActorState = Any
TrainState = Dict[str, Any]
Batch = Dict[
    str, jax.Array
]


ActionSpec = Any
ObservationSpec = Any


class RLAgent:
    """Abstract base class for a Reinforcement Learning agent.

    This class defines the core interface for an RL agent, including methods
    for initializing training and actor states, performing actor steps
    (selecting actions), and learner steps (updating parameters).
    Subclasses are expected to implement these methods.
    """

    def __init__(self) -> None:
        """Initializes the RLAgent."""
        raise NotImplementedError

    def initial_train_state(self, key: PRNGKey) -> TrainState:
        """Initializes the training state of the agent.

        This typically includes initializing network parameters and optimizer state.

        Args:
            key: A JAX PRNG key for random initialization.

        Returns:
            A dictionary representing the initial training state.
        """
        raise NotImplementedError

    def initial_actor_state(self) -> ActorState:
        """Initializes the state of the actor.

        This state is carried over between actor steps. For example, it could
        store the hidden state of a recurrent neural network.

        Returns:
            An initial actor state.
        """
        raise NotImplementedError

    def actor_step(
        self,
        train_state: TrainState,
        obs: Observation,
        actor_state: ActorState,
        key: PRNGKey,
        evaluation: bool,
    ) -> Tuple[Action, ActorState]:
        """Performs a single step of acting in the environment.

        Args:
            train_state: The current training state (e.g., network parameters).
            obs: The current observation from the environment.
            actor_state: The current actor state.
            key: A JAX PRNG key for any stochastic operations.
            evaluation: A boolean indicating whether the agent is in evaluation mode
                (e.g., to select actions greedily) or training mode.

        Returns:
            A tuple containing the selected action and the next actor state.
        """
        raise NotImplementedError

    def learner_step(
        self, train_state: TrainState, data: Batch, key: PRNGKey
    ) -> Tuple[TrainState, Loss]:
        """Performs a single step of learning/training.

        Args:
            train_state: The current training state.
            data: A batch of experience data for training (e.g., transitions).
            key: A JAX PRNG key for any stochastic operations.

        Returns:
            A tuple containing the updated training state and any metrics (e.g., loss).
        """
        raise NotImplementedError


class Random(RLAgent):
    """A simple RL agent that selects actions uniformly at random.

    This agent is often used as a baseline or for initial exploration.
    It does not perform any learning.
    """

    _action_spec: ActionSpec

    def __init__(self, action_spec: ActionSpec) -> None:
        """Initializes the Random agent.

        Args:
            action_spec: A specification of the action space, typically providing
                `num_values` for discrete action spaces.
        """
        self._action_spec = action_spec

    def initial_train_state(self, key: PRNGKey) -> TrainState:
        """Initializes the training state. For a Random agent, this is empty.

        Args:
            key: A JAX PRNG key (unused).

        Returns:
            An empty dictionary.
        """
        return dict()

    def initial_actor_state(self) -> ActorState:
        """Initializes the actor state. For a Random agent, this is empty.

        Returns:
            An empty tuple.
        """
        return ()

    def actor_step(
        self,
        train_state: TrainState, # Changed from params: Params to train_state: TrainState for consistency
        observation: Observation,
        actor_state: ActorState,
        key: PRNGKey,
        evaluation: bool,
    ) -> Tuple[Action, ActorState]:
        """Selects an action randomly from the available actions.

        Args:
            train_state: The current training state (unused by Random agent).
            observation: The current observation from the environment (unused).
            actor_state: The current actor state (unused).
            key: A JAX PRNG key for random action selection.
            evaluation: Evaluation mode flag (unused).

        Returns:
            A tuple containing a randomly selected action (integer) and the
            (unchanged) actor state.
        """
        del train_state, observation, evaluation # Unused by the Random agent
        action = jax.random.randint(key, (), 0, self._action_spec.num_values)
        return action, actor_state

    def learner_step(
        self, train_state: TrainState, data: Batch, key: PRNGKey
    ) -> Tuple[TrainState, Loss]:
        """Performs a learning step. For a Random agent, this does nothing.

        Args:
            train_state: The current training state.
            data: A batch of experience data (unused).
            key: A JAX PRNG key (unused).

        Returns:
            A tuple containing the (unchanged) training state and a loss of 0.0.
        """
        del data, key # Unused by the Random agent
        return train_state, jnp.array(0.0)


class QLearner(RLAgent):
    """An RL agent implementing the Q-learning algorithm.

    This agent learns a Q-value function, representing the expected return
    for taking an action in a given state. It uses a neural network to
    approximate the Q-function and updates it based on the Bellman equation.
    """

    _network: nn.Module # Changed from hk.Module to nn.Module (Flax)
    _batch_apply_net: Callable[[Params, Observation], jax.Array]
    _observation_spec: ObservationSpec
    _learning_rate: float
    _optimizer: optax.GradientTransformation
    _grad_clip_value: float = 1.0 # Added for clarity on clipping

    actor_step: Callable[
        [TrainState, Observation, ActorState, PRNGKey, bool], Tuple[Action, ActorState]
    ]
    learner_step: Callable[[TrainState, Batch, PRNGKey], Tuple[TrainState, Loss]]

    def __init__(
        self,
        network: nn.Module, # Changed from hk.Module
        observation_spec: ObservationSpec,
        learning_rate: float,
        grad_clip_value: float = 1.0,
    ) -> None:
        """Initializes the QLearner agent.

        Args:
            network: A Flax Linen network module (e.g., from rl_lib.utils.NN)
                that takes observations as input and outputs Q-values for each action.
            observation_spec: A specification of the observation space, used to
                generate dummy input for network initialization.
            learning_rate: The learning rate for the Adam optimizer.
            grad_clip_value: The value for global norm gradient clipping.
        """
        self._network = network
        # For Flax: network.apply takes {'params': params, ...}, obs
        # We define a helper for vmap if params are handled explicitly.
        def _apply_with_params(params: Params, obs: Observation) -> jax.Array:
            return self._network.apply({'params': params}, obs)
        self._batch_apply_net = jax.vmap(_apply_with_params, in_axes=(None, 0))

        self._observation_spec = observation_spec
        self._learning_rate = learning_rate
        self._grad_clip_value = grad_clip_value
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(self._grad_clip_value),
            optax.adam(learning_rate)
        )

        self.actor_step = jax.jit(self._uncompiled_actor_step)
        self.learner_step = jax.jit(self._uncompiled_learner_step)

    def initial_train_state(self, key: PRNGKey) -> TrainState:
        """Initializes the training state for the QLearner.

        This involves initializing the neural network parameters and the
        optimizer state.

        Args:
            key: A JAX PRNG key for initializing network parameters.

        Returns:
            A dictionary containing the initial 'params' (network parameters)
            and 'opt_state' (optimizer state).
        """
        sample_input: Observation = self._observation_spec.generate_value()
        # Flax init returns a dict of variables, e.g., {'params': ..., 'batch_stats': ...}
        # We assume the network's state is primarily 'params' for optimization.
        variables: PyTree = self._network.init(key, sample_input)
        params: Params = variables['params']
        opt_state: OptState = self._optimizer.init(params)
        return {"params": params, "opt_state": opt_state}

    def initial_actor_state(self) -> ActorState:
        """Initializes the actor state. For QLearner, this is typically empty.

        Returns:
            An empty tuple.
        """
        return ()

    def _uncompiled_actor_step(
        self,
        train_state: TrainState,
        obs: Observation,
        actor_state: ActorState,
        key: PRNGKey,
        evaluation: bool,
    ) -> Tuple[Action, ActorState]:
        """Selects an action based on the current Q-values. (Internal, pre-JIT version)"""
        params: Params = train_state["params"]
        # Flax apply: self._network.apply({'params': params, ...}, obs)
        q_values: jax.Array = self._network.apply({'params': params}, obs)

        train_a: Action = rlax.softmax().sample(key, q_values)
        eval_a: Action = rlax.greedy().sample(key, q_values)

        action: Action = jax.lax.select(evaluation, eval_a, train_a)
        return action, actor_state

    def actor_step(
        self,
        train_state: TrainState,
        obs: Observation,
        actor_state: ActorState,
        key: PRNGKey,
        evaluation: bool,
    ) -> Tuple[Action, ActorState]:
        return self._uncompiled_actor_step(
            train_state, obs, actor_state, key, evaluation
        )

    def _loss(
        self,
        params: Params,
        states: Observation,
        actions: Action,
        rewards: jax.Array,
        discounts: jax.Array,
        next_states: Observation,
        target_params: Optional[Params] = None, # target_params are also PyTree
    ) -> Loss:
        """Computes the Q-learning loss."""
        q_s: jax.Array = self._batch_apply_net(
            params, states
        )

        target_qs_online: jax.Array = self._batch_apply_net(params, next_states)
        target_qs_target: Optional[jax.Array] = None
        if target_params is not None:
            target_qs_target = self._batch_apply_net(target_params, next_states)

        final_target_qs = (
            target_qs_target if target_params is not None else target_qs_online
        )

        td_error: jax.Array = self._td_operator(
            q_s, actions, rewards, discounts, final_target_qs
        )
        # Regularization is applied to the 'params' PyTree
        return cost(td_error) + l2_regulariser(params)

    def _td_operator(
        self,
        q_tm1: jax.Array,
        a_t: Action,
        r_t: jax.Array,
        discount_t: jax.Array,
        q_t: jax.Array,
    ) -> jax.Array:
        """Computes the TD error for Q-learning."""
        q_learning_fn: Callable = jax.vmap(rlax.q_learning, in_axes=(0, 0, 0, 0, 0))
        return q_learning_fn(q_tm1, a_t, r_t, discount_t, q_t)

    def _uncompiled_learner_step(
        self, train_state: TrainState, batch: Batch, unused_key: PRNGKey
    ) -> Tuple[TrainState, Loss]:
        """Performs a single Q-learning update step. (Internal, pre-JIT version)"""
        # The loss function expects params as the first argument for jax.grad
        # It also needs other parts of the batch, including target_params if available.
        loss_fn_for_grad = lambda p: self._loss(p, **batch)
        grad_fn = jax.value_and_grad(loss_fn_for_grad)

        # Ensure 'target_params' from batch is correctly passed if it exists
        current_params = train_state["params"]
        loss, grads = grad_fn(current_params)

        # Gradient clipping is now part of the optimizer chain
        # clipped_grads: Params = optimizers.clip_grads(grads, self._grad_clip_value) # Removed
        updates, opt_state = self._optimizer.update(
            grads, train_state["opt_state"], current_params
        )
        new_params: Params = optax.apply_updates(current_params, updates)

        return {"params": new_params, "opt_state": opt_state}, loss

    def learner_step(
        self, train_state: TrainState, batch: Batch, unused_key: PRNGKey
    ) -> Tuple[TrainState, Loss]:
        return self._uncompiled_learner_step(train_state, batch, unused_key)


class QLambda(QLearner):
    """A Q-learning agent that uses eligibility traces with a given lambda."""

    _lambda: float

    def __init__(
        self,
        _lambda: float,
        network: nn.Module, # Changed from hk.Module
        observation_spec: ObservationSpec,
        learning_rate: float,
        grad_clip_value: float = 1.0,
    ) -> None:
        """Initializes the QLambda agent."""
        super().__init__(
            network=network,
            observation_spec=observation_spec,
            learning_rate=learning_rate,
            grad_clip_value=grad_clip_value,
        )
        self._lambda = _lambda

    def _loss( # Overrides QLearner._loss
        self,
        params: Params,
        states: Observation, # Expected (batch, sequence, obs_dims)
        actions: Action,    # Expected (batch, sequence, action_dims)
        rewards: jax.Array, # Expected (batch, sequence)
        discounts: jax.Array, # Expected (batch, sequence)
        next_states: Observation, # Expected (batch, sequence, obs_dims)
        target_params: Optional[Params] = None,
    ) -> Loss:
        """Computes the Q(lambda) loss for a batch of trajectories."""
        # For sequence data, network application needs to be mapped over sequence dimension.
        # _batch_apply_net already handles batch, so we map it over the sequence (dim 1 of states)
        # Assuming states are (batch_size, seq_len, *obs_shape)
        # And _batch_apply_net expects (batch_size, *obs_shape) -> (batch_size, num_actions)
        # We need to apply network to each step in sequence for each item in batch.
        
        # Let's assume _batch_apply_net is for non-sequential batch.
        # For sequential data, we vmap _apply_with_params over both batch and sequence.
        def _apply_seq(p, obs_seq): # obs_seq is (seq_len, *obs_shape)
            return jax.vmap(lambda obs_item: self._network.apply({'params': p}, obs_item))(obs_seq)

        # vmap over batch dimension
        batch_apply_seq_net = jax.vmap(_apply_seq, in_axes=(None, 0))

        q_s: jax.Array = batch_apply_seq_net(params, states) # (batch, seq, num_actions)

        target_qs_online: jax.Array = batch_apply_seq_net(params, next_states)
        final_target_qs: jax.Array
        if target_params is None:
            final_target_qs = target_qs_online
        else:
            final_target_qs = batch_apply_seq_net(target_params, next_states)

        td_error: jax.Array = self._td_operator(
            q_s, actions, rewards, discounts, final_target_qs
        )
        return cost(td_error) + l2_regulariser(params)

    def _td_operator( # Overrides QLearner._td_operator
        self,
        q_s: jax.Array, # (batch, seq, num_actions)
        a_s: Action,    # (batch, seq)
        r_s: jax.Array, # (batch, seq)
        discount_s: jax.Array, # (batch, seq)
        target_qs: jax.Array, # (batch, seq, num_actions)
    ) -> jax.Array:
        """Computes the TD error using rlax.q_lambda for sequences."""
        # rlax.q_lambda expects inputs for a single sequence.
        # We vmap it over the batch dimension.
        batch_q_lambda_fn: Callable = jax.vmap(
            rlax.q_lambda, in_axes=(0, 0, 0, 0, 0, None) # q_tm1, a_tm1, r_t, discount_t, q_t, lambda_
        )
        return batch_q_lambda_fn(q_s, a_s, r_s, discount_s, target_qs, self._lambda)


class SoftWatkins(QLambda):
    """A Q(lambda) agent using Soft Watkins Q-learning."""

    kappa: float

    def __init__(
        self,
        kappa: float,
        _lambda: float,
        network: nn.Module, # Changed from hk.Module
        observation_spec: ObservationSpec,
        learning_rate: float,
        grad_clip_value: float = 1.0,
    ) -> None:
        """Initializes the SoftWatkins agent."""
        super().__init__(
            _lambda=_lambda,
            network=network,
            observation_spec=observation_spec,
            learning_rate=learning_rate,
            grad_clip_value=grad_clip_value,
        )
        self.kappa = kappa

    def _td_operator( # Overrides QLambda._td_operator
        self,
        q_s: jax.Array,
        a_s: Action,
        r_s: jax.Array,
        discount_s: jax.Array,
        target_qs: jax.Array,
    ) -> jax.Array:
        """Computes the TD error using the soft Watkins operator."""
        # soft_watkins expects inputs for a single sequence.
        # We vmap it over the batch dimension.
        batch_soft_watkins_fn: Callable = jax.vmap(
            soft_watkins, in_axes=(0, 0, 0, 0, 0, None, None)
            # q_s, a_s, r_s, discount_s, target_qs, kappa, _lambda
        )
        return batch_soft_watkins_fn(
            q_s, a_s, r_s, discount_s, target_qs, self.kappa, self._lambda
        )


class EMATargetNet(SoftWatkins):
    """A SoftWatkins agent that uses an Exponential Moving Average (EMA) target network."""

    ema_fn: EMATree

    def __init__(
        self,
        ema_decay: float,
        kappa: float,
        _lambda: float,
        network: nn.Module, # Changed from hk.Module
        observation_spec: ObservationSpec,
        learning_rate: float,
        grad_clip_value: float = 1.0,
    ) -> None:
        """Initializes the EMATargetNet agent."""
        super().__init__(
            kappa=kappa,
            _lambda=_lambda,
            network=network,
            observation_spec=observation_spec,
            learning_rate=learning_rate,
            grad_clip_value=grad_clip_value,
        )
        self.ema_fn = EMATree(ema_decay)
        # Re-JIT learner_step because _uncompiled_learner_step is overridden
        self.learner_step = jax.jit(self._uncompiled_learner_step)


    def initial_train_state(self, key: PRNGKey) -> TrainState:
        """Initializes the training state, including the EMA state for target params."""
        state: TrainState = super().initial_train_state(key)
        # EMA state is a tree of params, initialized from the online network's params
        ema_target_params: Params = self.ema_fn.init(state["params"])
        return dict(**state, ema_target_params=ema_target_params)

    def _uncompiled_learner_step( # Overrides QLearner._uncompiled_learner_step
        self, train_state: TrainState, batch: Batch, unused_key: PRNGKey
    ) -> Tuple[TrainState, Loss]:
        """Performs a learner step, updating target network params with EMA."""
        # Current online parameters
        online_params = train_state["params"]
        # Current target parameters (from EMA state)
        # Note: ema_fn.init was used to initialize ema_target_params.
        # For update, ema_fn(current_ema_state, new_online_params)
        # Here, we want target_params for *this* step's loss calculation.
        # These should be the *previous* EMA params.
        current_target_params = train_state["ema_target_params"]

        # Prepare batch with these target_params for the loss function
        mutable_batch = dict(batch)
        mutable_batch["target_params"] = current_target_params

        # Define loss function for gradient calculation, using current online_params
        # and the (potentially stale) current_target_params from train_state.
        loss_fn_for_grad = lambda p: self._loss(p, **mutable_batch)
        grad_fn = jax.value_and_grad(loss_fn_for_grad)
        
        loss, grads = grad_fn(online_params)

        # Apply optimizer updates to online parameters
        updates, opt_state = self._optimizer.update(
            grads, train_state["opt_state"], online_params
        )
        new_online_params: Params = optax.apply_updates(online_params, updates)

        # Update the EMA target parameters using the *new* online parameters
        new_target_params: Params = self.ema_fn(
            current_target_params, new_online_params # state_tree, value_tree
        )
        
        final_updated_state = {
            "params": new_online_params,
            "opt_state": opt_state,
            "ema_target_params": new_target_params,
        }
        return final_updated_state, loss

    # learner_step is jitted in __init__
    def learner_step(
        self, train_state: TrainState, batch: Batch, unused_key: PRNGKey
    ) -> Tuple[TrainState, Loss]:
        return self._uncompiled_learner_step(train_state, batch, unused_key)


class MultiActionManager:
    """Abstract class for managing environments with multi-dimensional discrete actions."""
    action_spec: ActionSpec

    def __init__(self, action_spec: ActionSpec) -> None:
        self.action_spec = action_spec

    def iterate(self, nn_output: jax.Array) -> Iterator[jax.Array]:
        raise NotImplementedError

    def get_mask(self, action: Action) -> jax.Array:
        raise NotImplementedError


class MaskedMultiAction(EMATargetNet):
    """An agent for environments requiring multiple discrete actions per step.
    Note: The `run_loop` in `experiment.py` currently expects a single integer action.
    This agent's `actor_step` returns `List[Action]`, which may require modifications
    to `run_loop` or a wrapper around this agent for compatibility.
    """
    mam: MultiActionManager

    def __init__(
        self,
        mam: MultiActionManager,
        ema_decay: float,
        kappa: float,
        _lambda: float,
        network: nn.Module, # Changed from hk.Module
        observation_spec: ObservationSpec,
        learning_rate: float,
        grad_clip_value: float = 1.0,
    ) -> None:
        """Initializes the MaskedMultiAction agent."""
        super().__init__(
            ema_decay=ema_decay,
            kappa=kappa,
            _lambda=_lambda,
            network=network,
            observation_spec=observation_spec,
            learning_rate=learning_rate,
            grad_clip_value=grad_clip_value,
        )
        self.mam = mam
        # Re-JIT actor_step because _uncompiled_actor_step is overridden
        self.actor_step = jax.jit(self._uncompiled_actor_step)


    def _uncompiled_actor_step( # Overrides QLearner._uncompiled_actor_step
        self,
        train_state: TrainState,
        obs: Observation,
        actor_state: ActorState,
        key: PRNGKey,
        evaluation: bool,
    ) -> Tuple[List[Action], ActorState]: # Returns List of Actions
        """Selects a composite action by choosing each sub-action."""
        params: Params = train_state["params"]
        # Flax apply: self._network.apply({'params': params, ...}, obs)
        nn_output: jax.Array = self._network.apply({'params': params}, obs)

        chosen_actions: List[Action] = []
        current_key = key
        for q_values_head in self.mam.iterate(nn_output):
            current_key, subkey = jax.random.split(current_key)
            train_a: Action = rlax.softmax().sample(subkey, q_values_head)
            eval_a: Action = rlax.greedy().sample(subkey, q_values_head)
            action_part: Action = jax.lax.select(evaluation, eval_a, train_a)
            chosen_actions.append(action_part)

        return chosen_actions, actor_state

    # actor_step is jitted in __init__
    def actor_step(
        self,
        train_state: TrainState,
        obs: Observation,
        actor_state: ActorState,
        key: PRNGKey,
        evaluation: bool,
    ) -> Tuple[List[Action], ActorState]:
        return self._uncompiled_actor_step(
            train_state, obs, actor_state, key, evaluation
        )

    def _loss( # Overrides EMATargetNet -> SoftWatkins -> QLambda -> QLearner._loss
        self,
        params: Params,
        states: Observation, # (batch, seq, obs)
        actions: Action,     # (batch, seq, num_sub_actions)
        rewards: jax.Array,  # (batch, seq)
        discounts: jax.Array, # (batch, seq)
        next_states: Observation, # (batch, seq, obs)
        target_params: Optional[Params] = None,
    ) -> Loss:
        """Computes the loss for the multi-action setting."""
        # Similar to QLambda._loss, network application needs to handle sequences.
        def _apply_seq(p, obs_seq):
            return jax.vmap(lambda obs_item: self._network.apply({'params': p}, obs_item))(obs_seq)
        batch_apply_seq_net = jax.vmap(_apply_seq, in_axes=(None, 0))

        # q_s_raw will be (batch, seq, total_nn_output_dim)
        q_s_raw: jax.Array = batch_apply_seq_net(params, states)

        target_qs_raw_online: jax.Array = batch_apply_seq_net(params, next_states)
        final_target_qs_raw: jax.Array
        if target_params is None:
            final_target_qs_raw = target_qs_raw_online
        else:
            final_target_qs_raw = batch_apply_seq_net(target_params, next_states)

        total_loss: Loss = jnp.array(0.0)

        # actions is (batch, sequence, num_sub_actions)
        # self.mam.get_mask expects one sub_action of shape (num_sub_actions)
        # This seems to imply actions is not a composite action but one part of it.
        # Let's assume actions is (batch, sequence, num_parts_in_composite_action)
        # And mam.get_mask takes the full composite action for a step (num_parts_in_composite_action,)
        # and returns a mask for each head. This part is a bit ambiguous from original.
        # Assuming mam.get_mask(action_parts_for_one_step) -> list_of_masks_for_heads
        # The original code: mask_vmap_fn = jax.vmap(jax.vmap(self.mam.get_mask, in_axes=0), in_axes=0)
        # This implies self.mam.get_mask takes a single action part, not the composite action.
        # And actions[..., i] is used. This suggests actions is indeed (batch, seq, num_sub_actions)
        # where num_sub_actions is the number of heads.

        # Let's assume actions is (batch_size, sequence_length, num_action_heads)
        # And mask_per_action_type is (batch_size, sequence_length, num_action_heads)
        # where each entry is a mask for the TD error of that head.
        # The original mam.get_mask(action: Action) -> jax.Array implies it takes a single action value.
        # This needs clarification based on MultiActionManager's design.
        # For now, proceeding with the structure given.

        # If mam.get_mask is meant to operate on the *composite action* for a timestep
        # to produce masks for each head, the vmap structure would be different.
        # If mam.get_mask(action_component_i) -> mask_for_head_i, then it's as below.
        # The original code `mask_per_action_type[..., i]` suggests the latter.
        
        # Assuming self.mam.get_mask(action_from_one_head) -> mask_for_that_head_td_error
        # This seems unusual. A mask usually depends on the action *taken* to decide if a TD error is valid.
        # Let's assume the mask is derived from the action of the *current* head.
        mask_fn_head_wise = jax.vmap(jax.vmap(self.mam.get_mask, in_axes=0), in_axes=0) # map over batch and seq
        
        # q_s_raw and final_target_qs_raw are (batch, seq, total_nn_output_dim)
        # self.mam.iterate splits these into per-head Q-values.
        # Each q_head, target_q_head will be (batch, seq, num_actions_for_this_head)

        iter_q_s = list(self.mam.iterate(q_s_raw))
        iter_target_q_s = list(self.mam.iterate(final_target_qs_raw))

        for i in range(len(iter_q_s)):
            q_head = iter_q_s[i]
            target_q_head = iter_target_q_s[i]
            
            # actions_for_head_i is (batch, seq)
            actions_for_head_i: Action = actions[..., i]
            
            td_error_head: jax.Array = self._td_operator( # This is SoftWatkins._td_operator
                q_head, actions_for_head_i, rewards, discounts, target_q_head
            ) # td_error_head is (batch, seq)

            # Assuming get_mask is applied to the actions of the current head
            # to generate a mask for the TD errors of this head.
            # mask_for_head_i should be (batch, seq)
            mask_for_head_i: jax.Array = mask_fn_head_wise(actions_for_head_i)
            
            # Ensure mask_for_head_i is broadcastable with td_error_head
            if mask_for_head_i.ndim > td_error_head.ndim: # e.g. mask is (B,S,1) and td_error is (B,S)
                 mask_for_head_i = jnp.squeeze(mask_for_head_i, axis=-1)
            
            total_loss += cost(td_error_head * mask_for_head_i)

        return total_loss + l2_regulariser(params)