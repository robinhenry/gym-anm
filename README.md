[![codecov](https://codecov.io/gh/robinhenry/gym-anm/branch/master/graph/badge.svg?token=7JSMJPPIQ7)](https://codecov.io/gh/robinhenry/gym-anm)

**Status:** Active (under active development, breaking changes may occur).


# gym-anm

**gym-anm is an extension to the [OpenAI Gym](https://github.com/openai/gym) toolkit. It provides researchers 
with a tool to build and train reinforcement learning agents in environments 
modelling active network management tasks for electricity distribution 
networks.**
  <br>
  
[![Build Status](https://travis-ci.com/robinhenry/gym-anm.svg?token=y9jBzyzxgcYYHdKPqW9q&branch=master)](https://travis-ci.com/robinhenry/gym-anm)
  
Each `gym-anm` environment implements the same [general task](#general_task).
 
 This toolkit comes with a paper available at [ADD LINK], and 
 researchers should reference `gym-anm` with the BibTeX entry:
 ```bash
ADD BibTeX entry
```  

## Getting Started

These instructions will help you install `gym-anm` on your machine and get 
started.

### Prerequisites

A version of Python 3.5 or greater is required.

### Installing

You can install `gym-anm` with:
```
git clone https://github.com/robinhenry/gym-anm
cd gym-anm
pip install .
```

### Quick Start
You can now train your agents on `gym-anm` environments. For example:
```
import gym

env = gym.make('gym_anm:ANM6-Easy-v0')
o = env.reset()

for i in range(100):
    a = env.action_space.sample()
    o, r, _, _ = env.step(a)
```

## Authors

* **Robin Henry** - *Initial work* - [add personal website url]

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
