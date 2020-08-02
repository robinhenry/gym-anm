# -*- coding: utf-8 -*-
import torch
import numpy as np


class EnvMask():
    """docstring for EnvMask"""
    def __init__(self, env, period=2):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.period = period
        self.t = 0

    def step(self, action):
        """
        Corrupt the state resulting from `action` by setting it to zero except
        once every `self.period` steps.
        """
        self.t += 1
        y, r, d, info = self.env.step(action)
        if self.t % self.period == 0:
            return y, r, d, info
        return np.zeros(y.shape), r, d, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)


def check_u_size(u_low, u_high):
    """
    Check that the shape of the lower and upper bounds on the action dimensions
    corresponds and are of the form (u_size,), else raise a ValueError.

    Arguments
    ---------
     - u_low: array of shape (u_size,)
        Lower bounds on actions along each dimension of the action space
     - u_high: array of shape (u_size,)
        Upper bounds on actions along each dimension of the action space

    Returns
    -------
     - u_size: int
        The action space dimension common to `u_low` and `u_high`
     - u_low: tensor of shape (u_size,)
        Lower bounds on actions along each dimension of the action space
     - u_high: tensor of shape (u_size,)
        Upper bounds on actions along each dimension of the action space
    """
    if len(u_low) != len(u_high):
        raise ValueError('The lower and upper bounds should have the same '
                         'size corresponding to the action dimension')

    u_size = len(u_low)
    u_low = torch.Tensor(u_low)
    u_high = torch.Tensor(u_high)

    if u_low.dim() != 1 or u_high.dim() != 1:
        raise ValueError('The lower and upper bound should be of shape '
                         '(u_size,)')

    return u_size, u_low, u_high


def action_scaling(x, low, high):
    """
    Scale the input between the lower and upper bounds along each action
    dimension.

    Arguments
    ---------
     - x: tensor of shape (batch_size, u_dim)
        Input to scale, supposedly in [-1, 1]
     - low: tensor of shape (u_dim,)
        Lower bounds for output along each dimension
     - high: tensor of shape (u_dim,)
        Uppper bounds for output along each dimension
    """
    if low.dim() != 1 or high.dim() != 1:
        raise ValueError('Lower and upper bounds should have shape (u_dim,)')
    if x.size(-1) != low.size(0) or x.size(-1) != high.size(0):
        raise ValueError('Second dimension of the input should match lower and'
                         ' uppder bounds first dimensions')
    low = low.to(x)
    high = high.to(x)
    return .5 * (high - low) * x + .5 * (high + low)
