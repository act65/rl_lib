# RL Library

The goal of this project is to build an efficient and well-documented Reinforcement Learning (RL) library that can be used as a foundation for various RL projects and research. The library has recently been tidied, with significantly improved documentation, type hinting, and test coverage.

## Key Features & Implemented Components

This library provides a range of modular components for building RL agents and experiments:

### Learners (`rl_lib.learners`)
A collection of RL agent implementations:
-   **`Random`**: A simple agent that selects actions uniformly at random.
-   **`QLearner`**: Implements the standard Q-learning algorithm.
-   **`QLambda`**: Extends Q-learning with eligibility traces using the Q(lambda) algorithm.
-   **`SoftWatkins`**: A Q(lambda) agent that uses the 'soft watkins' TD update (from "Human-level Atari 200x faster") for off-policy corrections and multi-step returns.
-   **`EMATargetNet`**: A `SoftWatkins` agent that incorporates an Exponential Moving Average (EMA) for updating the target network, which can help stabilize training.
-   **`MaskedMultiAction`**: An agent designed for environments requiring multiple discrete actions per step, utilizing a `MultiActionManager` to handle structured action spaces.

### TD Operators (`rl_lib.td_operators`)
Core TD learning update rules:
-   **`soft_watkins`**: Implements the Soft Watkins Q(lambda) operator, allowing for dynamic lambda adjustment based on policy similarity.

### Accumulators/Replay Buffers (`rl_lib.accumulators`)
Tools for experience storage and replay:
-   **`OnlineAccumulator`**: A simple in-memory accumulator for single transitions, suitable for on-policy learning.
-   **`MultiStepOnlineAccumulator`**: An in-memory accumulator for N-step transition sequences.
-   **Reverb-based Accumulators**: Leveraging DeepMind's [Reverb](https://github.com/deepmind/reverb) library for efficient, off-policy replay:
    -   **`ReverbAccumulator`**: Standard Reverb integration supporting multi-step returns.
    -   **`MultiAgentReverbAccumulator`**: Extends `ReverbAccumulator` for multi-agent scenarios, writing data for multiple agents to the same Reverb table.
    -   **`RLPDAccumulator`**: Implements replay based on "Efficient Online Reinforcement Learning with Offline Data" (RLPD), managing 'online' and 'offline' data tables for mixed replay.
    -   **`MultiAgentRLPDAccumulator`**: A multi-agent version of `RLPDAccumulator`.

### Utilities (`rl_lib.utils`)
A collection of reusable helper functions and classes:
-   **`NN`**: A simple FeedForward Neural Network implemented in Haiku.
-   **`EMATree` & `EMA`**: Classes for applying Exponential Moving Average updates to PyTrees (e.g., network parameters).
-   **`build_signature`**: Utility for creating Reverb table signatures.
-   Custom cost functions and L2 regularization helpers.

## Running Experiments

Experiments are primarily run through the `experiments/main.py` script. It is configurable via command-line flags.

**Key Configuration Flags:**
-   `--environment_name`: Specifies the BSuite environment to use (e.g., "catch", "cartpole", "mountain_car", "bandit").
-   `--agent_name`: Determines the learning agent (e.g., "qlearner", "softwatkins", "ematargetnet").
-   `--accumulator_name`: Selects the experience replay mechanism (e.g., "reverb", "online", "multistep_online").
-   **Other important flags:**
    -   `--learning_rate`: Optimizer learning rate.
    -   `--batch_size`: Batch size for training updates.
    -   `--n_multistep`: Number of steps for N-step returns or Reverb trajectory length.
    -   `--train_episodes`: Total number of training episodes.
    -   `--seed`: Random seed for reproducibility.
    -   `--kappa`, `--_lambda`: Parameters for `SoftWatkins` and related agents.
    -   `--ema_decay`: EMA decay rate for `EMATargetNet`.
    -   `--reverb_port`: Port for the Reverb server if `ReverbAccumulator` is used.

**Example Command:**
```bash
python experiments/main.py --agent_name=ematargetnet --environment_name=cartpole --train_episodes=5000 --learning_rate=0.001 --n_multistep=3
```

## Development

The codebase includes comprehensive docstrings and type hints to aid understanding and development.

Unit tests are located in the `rl_lib/tests/` directory. They can be run using standard Python test runners. For example, to discover and run all tests:
```bash
python -m unittest discover -s rl_lib/tests
```
(Note: Ensure your Python environment is set up to find the `rl_lib` package, e.g., by running from the project root or having `rl_lib` in `PYTHONPATH`.)

## TODOs / Future Directions
-   **Uncertainty-based Exploration**: Planned support for incorporating uncertainty estimates into the learning process (e.g., via the `bellman_uncertainty` stub in `rl_lib.td_operators`).
-   Further evaluation and benchmarking of different agent and accumulator combinations.
-   Expansion of supported environments and agent types.

## Inspiration
Code is inspired in style by (and in some cases, adapted from) the [rlax](https://github.com/deepmind/rlax) examples provided by DeepMind.

## License
This project is licensed under the terms of the LICENSE file. (Please ensure a LICENSE file exists in the repository).
