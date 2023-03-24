import reverb
from functools import partial

import tensorflow as tf
import collections
import jax.numpy as jnp
import numpy as np

from rl_lib.utils import is_port_in_use

class Accumulator:
    def __init__(self, max_size):
        self.max_size = max_size
        self.names = ['states', 'actions', 'rewards', 'discounts', 'next_states']

    def push(self):
        raise NotImplementedError
    
    def sample(self, batch_size):
        raise NotImplementedError
    
    def is_ready(self, batch_size):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class OnlineAccumulator(Accumulator):
    """Simple Python accumulator for transitions."""

    def __init__(self, max_size=1):
        super().__init__(max_size=max_size)
        self.buffer = None
        self.prev_state = None

    def push(self, state, action, reward, discount):
        if action is None or self.prev_state is None:
            pass
        else:
            self.buffer = (self.prev_state, action, reward, discount, state)
        self.prev_state = state

    def sample(self, batch_size):
        assert batch_size == 1
        return {k: v for k, v in zip(self.names, self.buffer)}

    def is_ready(self, batch_size):
        assert batch_size == 1
        return self.buffer is not None
  
    def reset(self):
        self.buffer = None

class MultiStepOnlineAccumulator(OnlineAccumulator):
    """Simple Python accumulator for transitions."""

    def __init__(self, max_size=1, n_multistep=4):
        super().__init__(max_size=n_multistep)
        self.n_multistep = n_multistep
        self.buffer = collections.deque(maxlen=self.n_multistep)
        self.prev_state = None

    def push(self, state, action, reward, discount):
        if action is None or self.prev_state is None:
            pass
        else:
            self.buffer.append((self.prev_state, action, reward, discount, state))
        self.prev_state = state

    def sample(self, batch_size):
        assert batch_size == 1
        sample_data = [self.buffer[i] for i in range(self.n_multistep)]
        self.buffer.popleft()
        return {k: jnp.stack(e, axis=0)[None, ...] for k, e in zip(self.names, zip(*sample_data))}

    def is_ready(self, batch_size):
        assert batch_size == 1
        return len(self.buffer) >= self.n_multistep
    
    def reset(self):
        self.buffer.clear()

class ReplayWriter(Accumulator):
    def __init__(self, writer, table_name, n_multistep):
        super().__init__(max_size=None)
        self.writer = writer
        self.table_name = table_name
        self.n_multistep = n_multistep
        self.prev_state = None

    def push(self, state, action, reward, discount):
        if action is None or self.prev_state is None:
            pass
        else:
            eg = {k: v for k, v in zip(self.names, 
                (self.prev_state, np.array(action, dtype=np.int32), np.array(reward, dtype=np.float32), np.array(discount, dtype=np.float32), state))}
            self.writer.append(eg)

        self.prev_state = state

        if self.writer.episode_steps > self.n_multistep:
            data = {k: self.writer.history[k][-self.n_multistep:] for k in eg.keys()}

            self.writer.create_item(
                    table=self.table_name,
                    priority=1.0,
                    trajectory=data
                )
            
            self.writer.flush()
            
    def reset(self):
        self.writer.end_episode()

    def sample(self, batch_size):
        raise ValueError("ReplayWriter does not support sampling")

def new_table(name, signature, max_size=10000):
    return reverb.Table( 
        name=name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_size, 
        rate_limiter=reverb.rate_limiters.MinSize(1), 
        signature=signature)

def start_server(table_name, signature, checkpoint_path='/tmp/buffer.ckpt', port=None, max_size=10000):
    table = new_table(name=table_name, signature=signature, max_size=max_size)
    # checkpointer = reverb.checkpointers.DefaultCheckpointer(path=checkpoint_path)
    return reverb.Server(tables=[table], checkpointer=None, port=port)

def make_dataset_from_table(table_name, server_address, batch_size, shuffle_buffer_size):
    ds = reverb.TrajectoryDataset.from_table_signature(
        table=table_name,
        server_address=server_address,
        # max_in_flight_samples_per_worker: "A good rule of thumb is to set this value to 2-3x times the batch size used.""
        max_in_flight_samples_per_worker=batch_size*3)
    
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size*2)

    return ds

class ReverbAccumulator(Accumulator):
    def __init__(self, port, signature, max_size, n_multistep, batch_size=None, shuffle_buffer_size=None):
        super().__init__(max_size=max_size)
        self.batch_size = batch_size
        self.port = port
        self.shuffle_buffer_size = shuffle_buffer_size
        self.table_name = 'online'
        self.n_multistep = n_multistep
        self.server_address = f"localhost:{port}"
        self.signature = signature

        self.ds = None

        # id server is already running, dont try to start
        if not is_port_in_use(port):
            self.server = self._start_server()

        self.client = reverb.Client(self.server_address)
        self.replay_writer = self._new_writer(table_name=self.table_name)

    def _new_writer(self, table_name):
        return ReplayWriter(self.client.trajectory_writer(num_keep_alive_refs=self.n_multistep), table_name, self.n_multistep)

    def _start_server(self):
        return start_server(port=self.port, signature=self.signature, table_name=self.table_name, max_size=self.max_size)
        
    def _make_ds(self):
        return make_dataset_from_table(self.table_name, self.server_address, self.batch_size, shuffle_buffer_size=self.shuffle_buffer_size)

    def push(self, state, action, reward, discount):
        self.replay_writer.push(state, action, reward, discount)

    def sample(self, batch_size):
        assert batch_size == self.batch_size

        # delay constructing dataset until we want to sample
        if self.ds is None:
            self.ds = self._make_ds()

        batch = next(iter(self.ds.take(1)))
        return {k: jnp.array(v) for k, v in batch.data.items()}
    
    def is_ready(self, batch_size):
        return self.current_size >= batch_size
    
    @property
    def current_size(self):
        return self.client.server_info()[self.table_name].current_size
    
    def reset(self):
        self.replay_writer.reset()

class MultiAgentReverbAccumulator(ReverbAccumulator):
    """
    Use a different writer for each agent to keep their histories seperate.
    Assumes each unit id is unique per episode.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writers = dict()

    def push(self, state, action, reward, discount):
        for idx in action.keys():
            # maybe add unit_id to 
            if idx not in self.writers.keys():
                self.writers[idx] = self._new_writer()

            self.writers[idx].push(state[idx], action[idx], reward[idx], discount[idx])

    def reset(self):
        for idx in self.writers.keys():
            # TODO maybe we should del them?
            self.writers[idx].reset()

class RLPDAccumulator(ReverbAccumulator):
    """
    Implements [RLPD](https://arxiv.org/abs/2302.02948).
    Aka 50:50 offline:online replay.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # need a second writer. the ReverbAccumulator.replay_writer will push to the online table
        self.offline_writer = self._new_writer(table_name='offline')

    def push(self, state, action, reward, discount, table_name):
        if table_name == 'online':
            self.replay_writer.push(state, action, reward, discount)
        elif table_name == 'offline':
            self.offline_writer.push(state, action, reward, discount)
        else:
            raise ValueError(f"Unknown table name: {table_name}")

    def _make_ds(self):
        return build_rlpd_ds(self.server_address, self.batch_size, self.shuffle_buffer_size)

    def _start_server(self):
        return start_rlpd_server(self.signature, port=self.port, max_size=self.max_size)

    @property
    def current_size(self):
        return min(*self.status())
    
    @property
    def status(self):
        info = self.client.server_info()
        n_online = info['online'].current_size
        n_offline = info['offline'].current_size
        return n_online, n_offline
    
    def reset(self):
        self.replay_writer.reset()
        self.offline_writer.reset()

def start_rlpd_server(signature, checkpoint_path='/tmp/buffer.ckpt', port=None, max_size=10000):
    online_table = new_table(name='online', signature=signature, max_size=max_size)
    offline_table = new_table(name='offline', signature=signature, max_size=max_size)

    # checkpointer = reverb.checkpointers.DefaultCheckpointer(path=checkpoint_path)
    server = reverb.Server(tables=[offline_table, online_table], checkpointer=None, port=port)
    return server

def merge_ds(ds1, ds2):
    def concat(x, y):
        return {k: tf.concat([x.data[k], y.data[k]], axis=0) for k in x.data.keys()}
    ds = tf.data.Dataset.zip((ds1, ds2))
    ds = ds.map(concat)
    return ds

def build_rlpd_ds(server_address, batch_size, shuffle_buffer_size):
    """
    This ds should yield 50:50 offline:online data
    """
    make_ds = partial(make_dataset_from_table, 
        server_address=server_address, 
        batch_size=batch_size//2, 
        shuffle_buffer_size=shuffle_buffer_size)

    offline_ds = make_ds(table_name='offline',)
    online_ds = make_ds(table_name='online')
    
    return merge_ds(offline_ds, online_ds)

class MultiAgentRLPDAccumulator(RLPDAccumulator):
    """
    Finally. The accumulator I actually want to use.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writers = dict()

    def push(self, state, action, reward, discount, table_name):
        for idx in action.keys():
            if (idx, table_name) not in self.writers.keys():
                self.writers[(idx, table_name)] = self._new_writer(table_name)
            self.writers[(idx, table_name)].push(state[idx], action[idx], reward[idx], discount[idx])

    def reset(self):
        for idx in self.writers.keys():
            # TODO maybe we should del them?
            self.writers[idx].reset()