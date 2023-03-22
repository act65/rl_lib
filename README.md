The goal is to build an efficient learner which I can use for my other projects.

We use;
- the 'soft watkins' td update (from [Human-level Atari 200x faster](https://arxiv.org/abs/2209.07550)) to help correct for off policy actions and allow the use of multi step returns.
- an exponential moving average target network to help stabilise training (I havent seen elsewhere, but havent properly looked. still needs to be evaluated -- WIP)
- (TODOs) uncertainty + discount / exploration / multiagent / reward normalisation / etc

There are also some replay buffers implemented using [reverb](https://github.com/deepmind/reverb).
- a replay buffer supporting multi-step returns,
- a multi agent replay buffer,
- a replay buffer supporting offline / prior data (from [Efficient Online Reinforcement Learning with Offline Data](https://arxiv.org/abs/2302.02948))

Code is inspired in style (/ copied) by the [rlax](https://github.com/deepmind/rlax) examples.

