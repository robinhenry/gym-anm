# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch.distributions import Normal

from utils.gym import check_u_size, action_scaling


class FullyConnected(nn.Module):
    """
    Simple multilayer perceptron with free number and size of hidden layers.
    """

    def __init__(self, input_size, hidden_sizes, output_size=None):
        """
        Arguments
        ---------
         - input_size: int
            Input dimension
         - output_size: int
            Output dimension
         - hidden_sizes: iterable of int
            Number of hidden units in each hidden layer
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size else hidden_sizes[-1]

        # Hidden layers
        layers = []
        for h_size in hidden_sizes:
            layers.append(nn.Linear(input_size, h_size))
            # layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU(inplace=True))
            input_size = h_size

        # Output layer
        if output_size is not None:
            layers.append(nn.Linear(input_size, output_size))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        """
        Compute the forward pass in the sequential neural network.
        """
        output_size = x.size()[:-1] + (self.output_size,)
        return self.sequential(x.view(-1, self.input_size)).view(output_size)


class FullyConnectedCritic(FullyConnected):
    """
    Critic network using a multilayer perceptron.
    """

    def __init__(self, input_size, hidden_sizes):
        """
        Arguments
        ---------
         - input_size: int
            Input dimension
         - hidden_sizes: iterable of int
            Number of hidden units in each hidden layer
        """
        super().__init__(input_size, hidden_sizes, output_size=1)


class FullyConnectedActor(FullyConnected):
    """
    Actor network using a multilayer perceptron. Note that the output layer is
    an hyperbolic tangent rescaled along each dimension according to the lower
    and upper bounds of the action space.
    """

    def __init__(self, input_size, u_low, u_high, hidden_sizes):
        """
        Arguments
        ---------
         - input_size: int
            Input dimension
         - u_low: array of shape (u_size,)
            Lower bounds on actions along each dimension of the action space
         - u_high: array of shape (u_size,)
            Upper bounds on actions along each dimension of the action space
         - hidden_sizes: iterable of int
            Number of hidden units in each hidden layer
        """
        output_size, self.u_low, self.u_high = check_u_size(u_low, u_high)
        super().__init__(input_size, hidden_sizes, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Compute the forward pass through the multilayer perceptron. The output
        is scaled according to the lower and upper bounds
        """
        x = super().forward(x)
        x = self.tanh(x)
        return action_scaling(x, self.u_low, self.u_high)
