# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent
from networks.fully_connected import FullyConnectedActor, FullyConnectedCritic

from utils.memory import ReplayBuffer
from utils.noise import OrnsteinUhlenbeckNoise
from utils.gym import check_u_size
from itertools import chain


class DDPG(BaseAgent):
    """
    This agent is an implementation of the Deep Deterministic Policy Gradient
    algorithm. It uses fully connected neural networks as actor and critic, a
    replay buffer, and an Ornstein-Uhlenbeck noise process.
    """

    def __init__(self, env, gamma, render=False, cpu=False, rnn=None,
                 rnn_layers=2, rnn_h_size=256, seq_len=8, lr_actor=1e-4,
                 lr_critic=1e-3, tau=1e-2, buffer_capacity=2**16,
                 batch_size=32, theta=.1, sigma=1, hidden_sizes=(256, 256)):
        """
        Arguments
        ---------
         - env, gamma, render, cpu, rnn, rnn_layers, rnn_h_size
            See agents.base.BaseAgent
         - lr_actor: float
            Actor learning rate
         - lr_critic: float
            Critic learning rate
         - tau: float
            Polyak averaging rate
         - batch_size: int
            Batch size for training
         - buffer_capacity: int
            See utils.memory.ReplayBuffer
         - theta, sigma: float
            See utils.noise.OrnsteinUhlenbeckNoise
        """
        super().__init__(env, gamma, render, cpu, rnn, rnn_layers, rnn_h_size,
                         seq_len)

        # Actor and critic networks
        hx_size = self.x_size if rnn is None else rnn_h_size

        self.actor = FullyConnectedActor(
            hx_size, self.u_low, self.u_high, hidden_sizes).to(self.device)
        self.actor_tar = FullyConnectedActor(
            hx_size, self.u_low, self.u_high, hidden_sizes).to(self.device)
        self.critic = FullyConnectedCritic(
            hx_size + self.u_size, hidden_sizes).to(self.device)
        self.critic_tar = FullyConnectedCritic(
            hx_size + self.u_size, hidden_sizes).to(self.device)

        # Training parameters and optimizers
        self.tau = tau
        self.batch_size = batch_size

        self.optim_actor = optim.Adam(self.actor.parameters(), lr_actor)
        if self.rnn:
            self.optim_critic = optim.Adam(chain(self.critic.parameters(), self.rnn.parameters()), lr_critic)
        else:
            self.optim_critic = optim.Adam(self.critic.parameters(), lr_critic)

        # Initialization of target networks
        self.actor_tar.load_state_dict(self.actor.state_dict())
        self.critic_tar.load_state_dict(self.critic.state_dict())
        self.actor_tar.eval()
        self.critic_tar.eval()

        # Losses
        self.actor_losses = []
        self.critic_losses = []

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            self.u_low, self.u_high, theta, sigma)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, self.x_size, self.u_size,
                                   self.rnn.r_size if self.rnn else None,
                                   self.seq_len)

    def optimize(self):
        """
        See `agents.base.BaseAgent`.
        """
        # Sample a minibatch
        sequences_batch = self.buffer.sample(self.batch_size, return_xu=False)
        XY_seq, U_seq, R_seq, D_seq, H0 = sequences_batch

        # Put sequence dimension first and move to device
        XY_seq = XY_seq.to(self.device)
        U_seq = U_seq.to(self.device)
        R_seq = R_seq.to(self.device)
        D_seq = D_seq.to(self.device)
        H0 = H0.to(self.device) if H0 is not None else None

        if self.rnn:
            HXY_seq, _ = self.rnn(XY_seq, H0)
        else:
            HXY_seq, _ = XY_seq, None

        HX_seq = HXY_seq[:-1, ...]
        HY_seq = HXY_seq[1:, ...]

        targets = self.targets_(R_seq, HY_seq, D_seq)

        # Update the critic: towards the target (MSE)
        self.optim_critic.zero_grad()
        critic_output = self.critic(torch.cat((HX_seq, U_seq), dim=2))
        critic_loss = F.mse_loss(critic_output, targets)
        critic_loss.backward()
        self.optim_critic.step()

        # Update the actor: using the sampled policy gradient
        self.optim_actor.zero_grad()
        U_seqnew = self.actor(HX_seq.detach())
        actor_loss = - self.critic(torch.cat((HX_seq.detach(), U_seqnew), dim=2)).mean()
        actor_loss.backward()
        self.optim_actor.step()

        # Register statistics
        self.critic_losses.append(critic_loss.detach())
        self.actor_losses.append(actor_loss.detach())

        # Move the targets slowly
        self.poliak_averaging_()

    def step(self, x, h, train=False):
        """
        See `agents.base.BaseAgent`.
        """
        mode = self.actor.training
        self.actor.eval()

        if self.rnn:
            x_seq = x.unsqueeze(0).unsqueeze(0)
            z_seq, i = self.rnn(x_seq.to(self.device), h.to(self.device))
            z = z_seq.squeeze(0)
        else:
            z, i = x.unsqueeze(0).to(self.device), None

        # Choose action
        with torch.no_grad():
            u = self.actor(z).squeeze(0).cpu()
            if train:
                u += self.noise()
                u = torch.max(torch.min(u, self.u_high), self.u_low)

        # Play action
        y, r, d, _ = self.env.step(u.numpy())
        y = torch.tensor(y).float()

        transition = (x, h), u, r, (y, i), d

        # Render the game
        if self.render:
            self.env.render()

            self.actor.train(mode=mode)

        return transition

    def play(self, T, train=False):
        """
        See `agents.base.BaseAgent`.
        """
        self.actor.train(mode=train)
        self.critic.train(mode=train)

        # Initialize statistics
        cumulative_reward = 0.0
        trajectory_length = 0

        # Play the trajectory
        y, i = self.reset()
        t = 0
        for i_seq in range(T // self.seq_len):

            x_seq = torch.empty(self.seq_len, self.x_size)
            u_seq = torch.empty(self.seq_len, self.u_size)
            r_seq = torch.empty(self.seq_len, 1)
            d_seq = torch.empty(self.seq_len, 1, dtype=torch.bool)

            i = i.detach() if self.rnn else None
            h0 = i.squeeze(-2) if self.rnn else None  # remove batch dimension

            done = False
            for t_seq in range(self.seq_len):

                x, h = y, i
                (x, h), u, r, (y, i), d = self.step(x, h, train=train)

                cumulative_reward += r * (1 - d) * (self.gamma ** t)
                done = done or d

                x_seq[t_seq, :] = x
                u_seq[t_seq, :] = u
                r_seq[t_seq, :] = r
                d_seq[t_seq, :] = bool(d)

                t += 1

            if train:
                self.buffer.add(x_seq, u_seq, r_seq, y, d_seq, h0)

                # We wait for the hidden state to stabilize before training
                if i_seq > 0:
                    self.optimize()

            if done: break

        return cumulative_reward, t + 1

    def targets_(self, R, HY_seq, D):
        """
        Compute the target using one step boostrapping: R + gamma * Q(Y, V).

        Arguments
        ---------
        - R: tensor of shape (batch_size, 1)
            Batch of rewards
        - Y: tensor of shape (batch_size, x_size)
            Batch of next states
        - D: tensor of shape (batch_size, 1)
            Batch of done flags
        """
        with torch.no_grad():
            V = self.actor_tar(HY_seq)
            HY_V = torch.cat((HY_seq, V), dim=2)
            targets = R + (~ D) * self.gamma * self.critic_tar(HY_V)

        return targets

    def poliak_averaging_(self):
        """
        Update the agent's actor and critic target networks using poliak
        averaging.
        """
        for wct, wc in zip(self.critic_tar.parameters(),
                           self.critic.parameters()):
            wct.data.copy_(self.tau * wc + (1.0 - self.tau) * wct)

        for wat, wa in zip(self.actor_tar.parameters(),
                           self.actor.parameters()):
            wat.data.copy_(self.tau * wa + (1.0 - self.tau) * wat)
