"""This module provides a collection of tools for accumulating and replaying
agent experiences, commonly known as replay buffers or accumulators.

It includes:
- Base `Accumulator` class defining the interface for replay buffers.
- Simple in-memory accumulators like `OnlineAccumulator` and `MultiStepOnlineAccumulator`.
- Integration with Google's Reverb library for efficient, off-policy replay,
  including `ReverbAccumulator`, `MultiAgentReverbAccumulator`.
- Specialized accumulators like `RLPDAccumulator` and `MultiAgentRLPDAccumulator`
  that implement the "RLPD: Reinforcement Learning with Plannable Dynamics"
  strategy of mixing online and offline data.
- Helper functions for setting up Reverb servers and datasets.
"""

import reverb
from functools import partial
from typing import Any, Dict, List, Tuple, Optional, Deque, Union, Sequence

import tensorflow as tf
import collections
import jax.numpy as jnp
import numpy as np

from rl_lib.utils import is_port_in_use

# Type Aliases
Observation = Any  # Typically np.ndarray or jnp.ndarray
Action = Union[int, float, np.ndarray, jnp.ndarray]
Reward = float
Discount = float
TrajectoryItem = Tuple[Observation, Action, Reward, Discount, Observation]
SampledBatch = Dict[str, jnp.ndarray]
ReverbSignature = Union[Sequence[tf.TensorSpec], Dict[str, tf.TensorSpec]]


class Accumulator:
    """Abstract base class for an experience accumulator (replay buffer).

    Defines the common interface for pushing new experiences, sampling batches
    of experiences, checking readiness for sampling, and resetting the buffer.
    """

    max_size: Optional[int]
    names: List[str]

    def __init__(self, max_size: Optional[int]) -> None:
        """Initializes the Accumulator.

        Args:
            max_size: The maximum number of items the accumulator can store.
                Can be None for accumulators with no fixed size limit (e.g., Reverb writers).
        """
        self.max_size = max_size
        self.names = ["states", "actions", "rewards", "discounts", "next_states"]

    def push(
        self,
        state: Observation,
        action: Optional[Action],
        reward: Optional[Reward],
        discount: Optional[Discount],
        next_state: Observation,
        **kwargs: Any,
    ) -> None:  # next_state used by ReplayWriter, state by OnlineAccumulator
        """Pushes a new experience transition into the accumulator.

        Args:
            state: The state observed by the agent. (s_t or s_{t-1} depending on context)
            action: The action taken by the agent. (a_{t-1})
            reward: The reward received by the agent. (r_{t-1})
            discount: The discount factor for the transition. (d_{t-1})
            next_state: The next state observed by the agent. (s_t)
            **kwargs: Additional arguments specific to subclasses (e.g., table_name for RLPD).
        """
        raise NotImplementedError

    def sample(self, batch_size: int) -> SampledBatch:
        """Samples a batch of experiences from the accumulator.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A batch of experiences, typically a dictionary where keys are
            self.names and values are JAX/NumPy arrays.
        """
        raise NotImplementedError

    def is_ready(self, batch_size: int) -> bool:
        """Checks if the accumulator has enough data to sample a batch.

        Args:
            batch_size: The desired batch size for sampling.

        Returns:
            True if enough data is available, False otherwise.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the accumulator to its initial state.

        This typically involves clearing any stored experiences or resetting
        internal pointers.
        """
        raise NotImplementedError


class OnlineAccumulator(Accumulator):
    """A simple in-memory accumulator for single transitions.
    (Docstring from original)
    """

    buffer: Optional[TrajectoryItem]
    prev_state: Optional[Observation]

    def __init__(self, max_size: int = 1) -> None:
        """Initializes the OnlineAccumulator.
        (Docstring from original)
        """
        super().__init__(max_size=max_size)
        self.buffer = None
        self.prev_state = None

    def push(
        self,
        state: Observation,
        action: Optional[Action],
        reward: Optional[Reward],
        discount: Optional[Discount],
        next_state: Optional[Observation] = None,
    ) -> None:  # next_state is effectively `state` here
        """Pushes a transition. `state` is s_t, `prev_state` is s_{t-1}.
        (Docstring from original, adapted for clarity)
        """
        if (
            action is None
            or self.prev_state is None
            or reward is None
            or discount is None
        ):
            pass
        else:
            self.buffer = (self.prev_state, action, reward, discount, state)
        self.prev_state = state

    def sample(self, batch_size: int) -> SampledBatch:
        """Samples the single stored transition.
        (Docstring from original)
        """
        assert batch_size == 1, "OnlineAccumulator can only sample with batch_size=1"
        if self.buffer is None:
            raise ValueError("Buffer is empty, cannot sample.")
        # Values in buffer are directly used, assuming they are already jnp/np arrays if needed by consumer
        return {
            k: np.asarray(v) for k, v in zip(self.names, self.buffer)
        }  # Ensure values are array-like for consistency

    def is_ready(self, batch_size: int) -> bool:
        """Checks if a transition is available for sampling.
        (Docstring from original)
        """
        assert (
            batch_size == 1
        ), "OnlineAccumulator readiness check requires batch_size=1"
        return self.buffer is not None

    def reset(self) -> None:
        """Resets the accumulator by clearing the stored transition and previous state."""
        self.buffer = None
        self.prev_state = None


class MultiStepOnlineAccumulator(OnlineAccumulator):
    """An in-memory accumulator for N-step transitions.
    (Docstring from original)
    """

    n_multistep: int
    buffer: Deque[TrajectoryItem]  # Deque stores individual (s,a,r,d,s') tuples
    prev_state: Optional[Observation]

    def __init__(self, max_size: int = 1, n_multistep: int = 4) -> None:
        """Initializes the MultiStepOnlineAccumulator.
        (Docstring from original)
        """
        # max_size for super() is for the number of N-step sequences, not individual steps.
        # The internal buffer `self.buffer` uses `n_multistep` as its maxlen.
        super().__init__(max_size=n_multistep if max_size == 1 else max_size)
        self.n_multistep = n_multistep
        self.buffer = collections.deque(maxlen=self.n_multistep)
        self.prev_state = None

    def push(
        self,
        state: Observation,
        action: Optional[Action],
        reward: Optional[Reward],
        discount: Optional[Discount],
        next_state: Optional[Observation] = None,
    ) -> None:
        """Pushes a single step transition into the internal N-step buffer.
        (Docstring from original, adapted for clarity)
        """
        if (
            action is None
            or self.prev_state is None
            or reward is None
            or discount is None
        ):
            pass
        else:
            self.buffer.append((self.prev_state, action, reward, discount, state))
        self.prev_state = state

    def sample(self, batch_size: int) -> SampledBatch:
        """Samples the N-step sequence from the buffer.
        (Docstring from original)
        """
        assert batch_size == 1, "MultiStepOnlineAccumulator only supports batch_size=1"
        assert self.is_ready(batch_size), "Not enough data to sample N-step transition."

        sample_data_list: List[TrajectoryItem] = list(self.buffer)
        self.buffer.popleft()

        # Transposed data: e.g., all_states will be a list of N states
        transposed_data = zip(*sample_data_list)
        # Stack each component (states, actions, etc.) along axis 0, then add batch dim
        return {
            k: jnp.expand_dims(jnp.stack(np.asarray(e), axis=0), axis=0)
            for k, e in zip(self.names, transposed_data)
        }

    def is_ready(self, batch_size: int) -> bool:
        """Checks if `n_multistep` transitions are available.
        (Docstring from original)
        """
        assert (
            batch_size == 1
        ), "MultiStepOnlineAccumulator readiness check requires batch_size=1"
        return len(self.buffer) >= self.n_multistep

    def reset(self) -> None:
        """Resets the accumulator by clearing the buffer and previous state."""
        self.buffer.clear()
        self.prev_state = None


class ReplayWriter(Accumulator):
    """Wraps a Reverb TrajectoryWriter to push N-step sequences to a Reverb table.
    (Docstring from original)
    """

    writer: reverb.TrajectoryWriter
    table_name: str
    n_multistep: int
    prev_state: Optional[Observation]

    def __init__(
        self, writer: reverb.TrajectoryWriter, table_name: str, n_multistep: int
    ) -> None:
        """Initializes the ReplayWriter.
        (Docstring from original)
        """
        super().__init__(max_size=None)
        self.writer = writer
        self.table_name = table_name
        self.n_multistep = n_multistep
        self.prev_state = None

    def push(
        self,
        state: Observation,
        action: Optional[Action],
        reward: Optional[Reward],
        discount: Optional[Discount],
        next_state: Optional[Observation] = None,
    ) -> None:  # next_state is `state` here
        """Appends a single step to the Reverb writer's internal history.
        (Docstring from original, adapted for clarity)
        """
        if (
            action is None
            or self.prev_state is None
            or reward is None
            or discount is None
        ):
            pass
        else:
            single_step_data: Dict[str, Any] = {
                "states": self.prev_state,
                "actions": np.array(
                    action, dtype=np.int32
                ),  # Assuming discrete actions
                "rewards": np.array(reward, dtype=np.float32),
                "discounts": np.array(discount, dtype=np.float32),
                "next_states": state,
            }
            self.writer.append(single_step_data)

        self.prev_state = state

        if self.writer.episode_steps >= self.n_multistep:
            # history is Dict[str, Deque[Any]]
            trajectory_data: Dict[str, Any] = {
                k: list(self.writer.history[k])[-self.n_multistep :] for k in self.names
            }
            self.writer.create_item(
                table=self.table_name, priority=1.0, trajectory=trajectory_data
            )
            self.writer.flush()

    def reset(self) -> None:
        """Ends the current episode in the Reverb writer.
        (Docstring from original)
        """
        if (
            self.writer.episode_steps > 0
        ):  # Avoid ending empty episodes if reset is called multiple times
            self.writer.end_episode()
        self.prev_state = None

    def sample(self, batch_size: int) -> SampledBatch:
        """Sampling is not supported by ReplayWriter.
        (Docstring from original)
        """
        raise ValueError(
            "ReplayWriter does not support sampling. Sample from Reverb table directly."
        )

    def is_ready(self, batch_size: int) -> bool:
        """Readiness is determined by the Reverb table, not this writer.
        (Docstring from original)
        """
        return False


def new_table(
    name: str, signature: ReverbSignature, max_size: int = 10000
) -> reverb.Table:
    """Creates a new Reverb table with uniform sampling and FIFO removal.
    (Docstring from original)
    """
    return reverb.Table(
        name=name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_size,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=signature,
    )


def start_server(
    table_name: str,
    signature: ReverbSignature,
    checkpoint_path: Optional[str] = "/tmp/buffer.ckpt",
    port: Optional[int] = None,
    max_size: int = 10000,
) -> reverb.Server:
    """Starts a Reverb server with a single table.
    (Docstring from original)
    """
    table = new_table(name=table_name, signature=signature, max_size=max_size)
    actual_checkpointer: Optional[reverb.checkpointers.CheckpointerBase] = None
    if checkpoint_path:
        # checkpointer = reverb.checkpointers.DefaultCheckpointer(path=checkpoint_path) # Example
        pass  # Checkpointer currently disabled in original code
    return reverb.Server(tables=[table], checkpointer=actual_checkpointer, port=port)


def make_dataset_from_table(
    table_name: str, server_address: str, batch_size: int, shuffle_buffer_size: int
) -> tf.data.Dataset:
    """Creates a TensorFlow Dataset from a Reverb table for consumption.
    (Docstring from original)
    """
    ds = reverb.TrajectoryDataset.from_table_signature(
        table=table_name,
        server_address=server_address,
        max_in_flight_samples_per_worker=batch_size * 3,
    )
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class ReverbAccumulator(Accumulator):
    """An accumulator that uses a Reverb server for experience replay.
    (Docstring from original)
    """

    batch_size: Optional[int]
    port: int
    shuffle_buffer_size: Optional[int]
    table_name: str
    n_multistep: int
    server_address: str
    signature: ReverbSignature
    ds: Optional[tf.data.Dataset]
    server: Optional[reverb.Server]
    client: reverb.Client
    replay_writer: ReplayWriter

    def __init__(
        self,
        port: int,
        signature: ReverbSignature,
        max_size: int,
        n_multistep: int,
        batch_size: Optional[int] = None,
        shuffle_buffer_size: Optional[int] = None,
        start_server_flag: bool = True,
    ) -> None:
        """Initializes the ReverbAccumulator.
        (Docstring from original)
        """
        super().__init__(max_size=max_size)
        self.batch_size = batch_size
        self.port = port
        self.shuffle_buffer_size = shuffle_buffer_size
        self.table_name = "online"
        self.n_multistep = n_multistep
        self.server_address = f"localhost:{port}"
        self.signature = signature

        self.ds = None
        self.server = None

        if start_server_flag and not is_port_in_use(port):
            self.server = self._start_server()

        self.client = reverb.Client(self.server_address)
        self.replay_writer = self._new_writer(table_name=self.table_name)

    def _new_writer(self, table_name: str) -> ReplayWriter:
        """Creates a new ReplayWriter for the specified table."""
        trajectory_writer: reverb.TrajectoryWriter = self.client.trajectory_writer(
            num_keep_alive_refs=self.n_multistep
        )
        return ReplayWriter(trajectory_writer, table_name, self.n_multistep)

    def _start_server(self) -> reverb.Server:
        """Starts a Reverb server with the 'online' table."""
        return start_server(
            port=self.port,
            signature=self.signature,
            table_name=self.table_name,
            max_size=self.max_size,
        )

    def _make_ds(self) -> tf.data.Dataset:
        """Creates a tf.data.Dataset for sampling from the 'online' table."""
        if self.batch_size is None or self.shuffle_buffer_size is None:
            raise ValueError(
                "batch_size and shuffle_buffer_size must be set to create a dataset."
            )
        return make_dataset_from_table(
            self.table_name,
            self.server_address,
            self.batch_size,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )

    def push(
        self,
        state: Observation,
        action: Optional[Action],
        reward: Optional[Reward],
        discount: Optional[Discount],
        next_state: Optional[Observation] = None,
    ) -> None:  # next_state is effectively `state` for ReplayWriter
        """Pushes a single step transition via the internal ReplayWriter.
        (Docstring from original, adapted for clarity)
        """
        self.replay_writer.push(
            state, action, reward, discount, next_state=state
        )  # Pass state as next_state to ReplayWriter

    def sample(self, batch_size: int) -> SampledBatch:
        """Samples a batch of N-step trajectories from the Reverb table.
        (Docstring from original)
        """
        assert (
            self.batch_size is not None and batch_size == self.batch_size
        ), f"Sample batch_size ({batch_size}) must match ReverbAccumulator.batch_size ({self.batch_size})"

        if self.ds is None:
            self.ds = self._make_ds()

        batch_sample_info: reverb.SampleInfo = next(iter(self.ds.take(1)))
        return {
            k: jnp.array(v.numpy()) for k, v in batch_sample_info.data.items()
        }  # Convert TF tensors to JAX arrays

    def is_ready(self, batch_size: int) -> bool:
        """Checks if the Reverb table has enough items to sample a batch.
        (Docstring from original)
        """
        return self.current_size >= batch_size

    @property
    def current_size(self) -> int:
        """Returns the current number of items in the 'online' Reverb table."""
        try:
            server_info = self.client.server_info()
            table_info = server_info.get(self.table_name)
            return table_info.current_size if table_info else 0
        except Exception:
            return 0

    def reset(self) -> None:
        """Resets the internal ReplayWriter, ending its current episode."""
        self.replay_writer.reset()


class MultiAgentReverbAccumulator(ReverbAccumulator):
    """A ReverbAccumulator for multi-agent scenarios.
    (Docstring from original)
    """

    writers: Dict[Any, ReplayWriter]  # Agent ID as key

    def __init__(
        self,
        port: int,
        signature: ReverbSignature,
        max_size: int,
        n_multistep: int,
        batch_size: Optional[int] = None,
        shuffle_buffer_size: Optional[int] = None,
        start_server_flag: bool = True,
    ) -> None:
        """Initializes the MultiAgentReverbAccumulator.
        (Docstring from original)
        """
        super().__init__(
            port,
            signature,
            max_size,
            n_multistep,
            batch_size,
            shuffle_buffer_size,
            start_server_flag,
        )
        self.writers = {}

    def push(
        self,
        state: Dict[Any, Observation],
        action: Dict[Any, Optional[Action]],
        reward: Dict[Any, Optional[Reward]],
        discount: Dict[Any, Optional[Discount]],
        next_state: Optional[Dict[Any, Observation]] = None,
    ) -> None:  # next_state is effectively `state` for ReplayWriter
        """Pushes transitions for multiple agents.
        (Docstring from original, adapted for clarity)
        """
        for idx in action.keys():
            if idx not in self.writers:
                self.writers[idx] = self._new_writer(self.table_name)

            # Ensure all components for this agent are present if action is not None
            if action[idx] is not None:
                self.writers[idx].push(
                    state[idx],
                    action[idx],
                    reward[idx],
                    discount[idx],
                    next_state=state[idx],
                )

    def reset(self) -> None:
        """Resets all individual agent writers and clears the writers dictionary."""
        for idx in list(
            self.writers.keys()
        ):  # Iterate over a copy of keys if modifying dict
            self.writers[idx].reset()
        self.writers = {}  # Re-initialize to an empty dictionary


class RLPDAccumulator(ReverbAccumulator):
    """Implements RLPD (Reinforcement Learning with Plannable Dynamics) replay.
    (Docstring from original)
    """

    offline_writer: ReplayWriter

    def __init__(
        self,
        port: int,
        signature: ReverbSignature,
        max_size: int,
        n_multistep: int,
        batch_size: Optional[int] = None,
        shuffle_buffer_size: Optional[int] = None,
        start_server_flag: bool = True,
    ) -> None:
        """Initializes the RLPDAccumulator.
        (Docstring from original)
        """
        super().__init__(
            port,
            signature,
            max_size,
            n_multistep,
            batch_size,
            shuffle_buffer_size,
            start_server_flag,
        )
        self.offline_writer = self._new_writer(table_name="offline")

    def push(
        self,
        state: Observation,
        action: Optional[Action],
        reward: Optional[Reward],
        discount: Optional[Discount],
        table_name: str,
        next_state: Optional[Observation] = None,
    ) -> None:  # next_state is effectively `state` for ReplayWriter
        """Pushes a transition to the specified table ('online' or 'offline').
        (Docstring from original, adapted for clarity)
        """
        if table_name == "online":
            self.replay_writer.push(state, action, reward, discount, next_state=state)
        elif table_name == "offline":
            self.offline_writer.push(state, action, reward, discount, next_state=state)
        else:
            raise ValueError(
                f"Unknown table name: {table_name}. Must be 'online' or 'offline'."
            )

    def _make_ds(self) -> tf.data.Dataset:
        """Creates a tf.data.Dataset that samples 50:50 from 'online' and 'offline' tables."""
        if self.batch_size is None or self.shuffle_buffer_size is None:
            raise ValueError(
                "batch_size and shuffle_buffer_size must be set to create RLPD dataset."
            )
        return build_rlpd_ds(
            self.server_address, self.batch_size, self.shuffle_buffer_size
        )

    def _start_server(self) -> reverb.Server:
        """Starts a Reverb server with both 'online' and 'offline' tables."""
        return start_rlpd_server(self.signature, port=self.port, max_size=self.max_size)

    def sample(self, batch_size: int) -> SampledBatch:
        """Samples a mixed batch (50:50) from 'online' and 'offline' tables.
        (Docstring from original)
        """
        assert (
            self.batch_size is not None and batch_size == self.batch_size
        ), f"Sample batch_size ({batch_size}) must match RLPDAccumulator.batch_size ({self.batch_size})"

        if self.ds is None:
            self.ds = self._make_ds()

        batch_sample_info_tuple: Tuple[reverb.SampleInfo, reverb.SampleInfo] = next(iter(self.ds.take(1)))  # type: ignore

        # build_rlpd_ds now yields merged SampleInfo directly due to merge_ds change
        # So, batch_sample_info_tuple is actually a single SampleInfo object with merged data
        merged_batch_sample_info: reverb.SampleInfo = batch_sample_info_tuple  # type: ignore

        return {
            k: jnp.array(v.numpy()) for k, v in merged_batch_sample_info.data.items()
        }

    @property
    def current_size(self) -> int:
        """Returns the minimum size of the 'online' and 'offline' tables.
        (Docstring from original)
        """
        return min(*self.status)

    @property
    def status(self) -> Tuple[int, int]:
        """Returns the current number of items in 'online' and 'offline' tables.
        (Docstring from original)
        """
        try:
            info = self.client.server_info()
            # Ensure robust access to current_size, defaulting to 0 if table not found
            n_online = info.get("online", reverb.TableInfo(name="online", current_size=0, signature=self.signature)).current_size  # type: ignore
            n_offline = info.get("offline", reverb.TableInfo(name="offline", current_size=0, signature=self.signature)).current_size  # type: ignore
            return n_online, n_offline
        except Exception:
            return 0, 0

    def reset(self) -> None:
        """Resets both the 'online' and 'offline' ReplayWriters."""
        self.replay_writer.reset()
        self.offline_writer.reset()


def start_rlpd_server(
    signature: ReverbSignature,
    checkpoint_path: Optional[str] = "/tmp/buffer.ckpt",
    port: Optional[int] = None,
    max_size: int = 10000,
) -> reverb.Server:
    """Starts a Reverb server with 'online' and 'offline' tables for RLPD.
    (Docstring from original)
    """
    online_table = new_table(name="online", signature=signature, max_size=max_size)
    offline_table = new_table(name="offline", signature=signature, max_size=max_size)

    actual_checkpointer: Optional[reverb.checkpointers.CheckpointerBase] = None
    if checkpoint_path:
        # actual_checkpointer = reverb.checkpointers.DefaultCheckpointer(path=checkpoint_path) # Example
        pass
    server = reverb.Server(
        tables=[offline_table, online_table],
        checkpointer=actual_checkpointer,
        port=port,
    )
    return server


def merge_ds(ds1: tf.data.Dataset, ds2: tf.data.Dataset) -> tf.data.Dataset:
    """Merges two TensorFlow Datasets yielding Reverb SampleInfo objects.
    (Docstring from original)
    """

    def concat_sample_infos(
        sample_info1: reverb.SampleInfo, sample_info2: reverb.SampleInfo
    ) -> reverb.SampleInfo:
        concatenated_data: Dict[str, tf.Tensor] = {  # type: ignore
            k: tf.concat([sample_info1.data[k], sample_info2.data[k]], axis=0)
            for k in sample_info1.data.keys()
        }
        return reverb.SampleInfo(
            key=sample_info1.key,
            probability=sample_info1.probability,  # type: ignore
            table=sample_info1.table,
            data=concatenated_data,
        )  # type: ignore

    ds = tf.data.Dataset.zip((ds1, ds2))
    ds = ds.map(concat_sample_infos)
    return ds


def build_rlpd_ds(
    server_address: str, batch_size: int, shuffle_buffer_size: int
) -> tf.data.Dataset:
    """Builds a dataset for RLPD that yields 50:50 online:offline data.
    (Docstring from original)
    """
    if batch_size % 2 != 0:
        raise ValueError("batch_size for RLPD must be even for 50:50 split.")

    half_batch_size = batch_size // 2

    make_half_ds_partial = partial(
        make_dataset_from_table,
        server_address=server_address,
        batch_size=half_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    offline_ds: tf.data.Dataset = make_half_ds_partial(table_name="offline")
    online_ds: tf.data.Dataset = make_half_ds_partial(table_name="online")

    return merge_ds(offline_ds, online_ds)


class MultiAgentRLPDAccumulator(RLPDAccumulator):
    """A multi-agent version of `RLPDAccumulator`.
    (Docstring from original)
    """

    writers: Dict[Tuple[Any, str], ReplayWriter]  # Key is (agent_id, table_name)

    def __init__(
        self,
        port: int,
        signature: ReverbSignature,
        max_size: int,
        n_multistep: int,
        batch_size: Optional[int] = None,
        shuffle_buffer_size: Optional[int] = None,
        start_server_flag: bool = True,
    ) -> None:
        """Initializes the MultiAgentRLPDAccumulator.
        (Docstring from original)
        """
        super().__init__(
            port,
            signature,
            max_size,
            n_multistep,
            batch_size,
            shuffle_buffer_size,
            start_server_flag,
        )
        self.writers = {}

    def push(
        self,
        state: Dict[Any, Observation],
        action: Dict[Any, Optional[Action]],
        reward: Dict[Any, Optional[Reward]],
        discount: Dict[Any, Optional[Discount]],
        table_name: str,
        next_state: Optional[Dict[Any, Observation]] = None,
    ) -> None:  # next_state is effectively `state`
        """Pushes transitions for multiple agents to the specified table.
        (Docstring from original, adapted for clarity)
        """
        for idx in action.keys():
            writer_key = (idx, table_name)
            if writer_key not in self.writers:
                self.writers[writer_key] = self._new_writer(table_name)

            if (
                action[idx] is not None
            ):  # Ensure there's an action to push for this agent
                self.writers[writer_key].push(
                    state[idx],
                    action[idx],
                    reward[idx],
                    discount[idx],
                    next_state=state[idx],
                )

    def reset(self) -> None:
        """Resets all individual agent/table writers and clears the writers dictionary."""
        for writer_key in list(self.writers.keys()):  # Iterate over a copy of keys
            self.writers[writer_key].reset()
        self.writers = {}  # Re-initialize to an empty dictionary
