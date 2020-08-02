# -*- coding: utf-8 -*-
# This module provides an Abstract Class for a RL Agent. New RL Agents only
# have to implement the abstract methods (step, optimize, play) and may
# eventually overwrite the save and load methods. It is adviced to call the
# constructor of BaseAgent in the child classes.
import pickle
import torch
import gym

import networks

from abc import ABC, abstractmethod
from pathlib import Path

from utils.gym import check_u_size


class BaseAgent(ABC):
    """
    Agent is an Abstract Class for a Reinforcement Learning agent suited for
    the continuous gym environments <https://github.com/openai/gym>.

    The Reinforcement Learning notations used in this module is `x(h)ury(i)d`
    (state - (hidden) - action - reward - next state - (next hidden) - done).
    """

    def __init__(self, env, gamma, render=False, cpu=False, rnn=None,
                 rnn_layers=2, rnn_h_size=256, seq_len=8):
        """
        Arguments
        ---------
         - env: gym.Env
            RL environment in which the agent plays
         - gamma: float
            Discount factor
         - render: bool
            Whether to render the environment at each time step
         - cpu: bool
            Whether to force working on the CPU only
         - rnn: str
            Type of rnn used between the environment and the agent as memory.
         - rnn_layers: int
            Number of layers in the rnn
         - rnn_h_size: int
            Number of hidden units in the hidden states at each layer
         - seq_len: int
            If rnn is not None, the length of sequences on which to train
        """
        self.env = env
        self.gamma = gamma
        self.render = render

        self.rnn_layers = rnn_layers
        self.rnn_h_size = rnn_h_size
        self.seq_len = seq_len

        # State and action space sizes
        self.x_size = env.observation_space.shape[0]
        u_low = env.action_space.low
        u_high = env.action_space.high
        self.u_size, self.u_low, self.u_high = check_u_size(u_low, u_high)

        # Set CUDA as device if available
        if not cpu and torch.cuda.is_available():
            print('CUDA available, moving to GPU')
            self.device = torch.device('cuda')
        else:
            print('CUDA unavailable, work on CPU')
            self.device = torch.device('cpu')

        # Instantiate memory network
        if rnn is not None:
            RNN = networks.rnns[rnn]
            self.rnn = RNN(input_size=self.x_size, hidden_size=rnn_h_size,
                           num_layers=rnn_layers)
            self.rnn.to(self.device)
        else:
            self.rnn = None
            self.seq_len = 1

    def reset(self):
        """
        The agent resets itself and the memory for a new trajectory.

        Return
        ------
         - x0: tensor of shape (x_dim,)
            New initial state
         - h0: tensor of shape ([...,] num_layers, 1, rnn_h_size) where 1
               corresponds to the batch_size
            New initial hidden state (the first dimensions are kept to
            structure the several hidden states (e.g. LSTM has (2, num_layers)
            for storing the `c` and `h` states for each layers))
        """
        x0 = torch.tensor(self.env.reset()).float()
        # The penultimate dimension is unsqueezed because it is the batch dim.
        h0 = torch.zeros(self.rnn.r_size).unsqueeze(-2) if self.rnn else None
        return x0, h0

    @abstractmethod
    def optimize(self):
        """
        The agent perform one optimization step (e.g. gradient descent step).
        Typically, this would be called after each `self.step()` call
        """
        pass

    @abstractmethod
    def step(self, x, h, train=False):
        """
        The agent takes one step in the environment according to the state `x`.

        Arguments
        ---------
         - x: tensor of shape (x_dim,)
            Current state of the environment
         - h: tensor of shape ([...,] num_layers, 1, rnn_h_size) where 1
               corresponds to the batch_size
            Current hidden state of the memory
         - train: bool
            Whether the agent is being trained in this step

        Returns
        -------
        (x, h), u, r, (y, i), d
         - (x, h)
             - x: tensor of shape (x_size,)
             - h: tensor of shape (..., 1, hidden_size)
            Current Memory state
         - u: tensor of shape (u_dim,)
            Action taken by the agent
         - r: float
            Reward obtained by the agent
         - (y, i)
             - y: tensor of shape (y_size,)
             - h: tensor of shape (..., 1, hidden_size)
            Next memory state
         - d: bool
            Whether the environment has reached a terminal state
        """
        pass

    @abstractmethod
    def play(self, T, train=False):
        """
        The agent play a new trajectory.

        Arguments
        ---------
         - T: int
            Maximum length of the trajectory
         - train: bool
            Whether to train the agent along the trajectory

        Returns
        -------
         - cumulative_reward
            Total discounted reward on the trajectory
         - trajectory_length
            Length of the trajectory
        """
        pass

    def eval(self, T):
        """
        The agent play a new trajectory without training on it. Equivalent to
        `Agent.play(self, T, train=False)`.
        """
        return self.play(T, train=False)

    def train(self, T):
        """
        The agent play a new trajectory and train itself on it. Equivalent to
        `Agent.play(self, T, train=True)`.
        """
        return self.play(T, train=True)

    def save(self, filepath):
        """
        Save the current state of the agent as is, using pickle. Child class
        should implement a more sparse `save` method.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    def load(self, filepath):
        """
        Load an agent state from a pickle file. If `save` mathod is changed,
        this function should be changed accordingly.
        """
        with open(filepath, 'rb') as file:
            self = pickle.load(file)
