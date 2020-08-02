# -*- coding: utf-8 -*-
import torch

from torch.distributions import MultivariateNormal

from utils.gym import check_u_size


# Inspired by the class OUNoise from <https://github.com/activatedgeek/torchrl>
class OrnsteinUhlenbeckNoise():
    """
    OrnsteinUhlenbeckNoise is the implementation of a class perturbing actions
    according to an Ornstein-Uhlenbeck process, where the noise is drawn from a
    (multivariate) normal distribution.
    """

    def __init__(self, u_low, u_high, theta, sigma, mu=0.0, reset_interval=50):
        """
        Parameters
        ----------
         - u_low: array of shape (u_size,)
            Lower bound on actions along each dimension of the action space
         - u_high: array of shape (u_size,)
            Upper bound on actions along each dimension of the action space
         - theta: float
            The first parameter of the Ornstein-Uhlenbeck process
         - sigma: float
            The second parameter of the Ornstein-Uhlenbeck process
         - mu: float
            The third parameter of the Ornstein-Uhlenbeck process
         - reset_interval: integer
            The number of steps after which to reset the noisy process
        """
        self.u_size, self.u_low, self.u_high = check_u_size(u_low, u_high)

        self.theta = theta
        self.sigma = sigma
        self.mu = mu

        # Random normal noise
        self.normal = MultivariateNormal(torch.zeros(self.u_size),
                                         torch.eye(self.u_size))

        # Process state
        self.reset_noise()
        self.t = 0
        self.reset_interval = reset_interval

    def reset_noise(self):
        """
        Reset the noise process.
        """
        self.x = self.normal.sample() * (self.u_high - self.u_low) / 16

    def __call__(self):
        """
        Compute a noise to be added to a certain action.

        Returns
        -------
         - noise: tensor of shape (u_dim)
            Noise resulting from the process
        """
        if self.t % self.reset_interval == 0:
            self.reset_noise()

        # Compute next noise value
        dx = (self.theta * (self.mu - self.x) +
              self.sigma * self.normal.sample())
        self.x += dx
        self.t += 1

        return self.x
