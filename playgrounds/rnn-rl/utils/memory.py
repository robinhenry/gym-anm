# -*- coding: utf-8 -*-
import torch


class ReplayBuffer():
    """
    ReplayBuffer stores transitions from trajectories. New transitions can be
    added to the buffer and can be sampled from it.
    """

    def __init__(self, capacity, x_size, u_size, r_size=None, seq_len=1):
        """
        Arguments
        ---------
         - capacity: int
            Number of transitions that can be stored
         - x_size: int
            Dimension of the state space
         - u_size: int
            Dimension of the action space
         - r_size: iterable or None
            Dimension of the recurrent state if using RNN, else None. Dimension
            under the format (num_layers, hidden_size)
         - seq_len: int
            Length of the sequences stored
        """
        self.seq_capacity = capacity // seq_len
        self.seq_len = seq_len

        # Storage for the sequences (nb: x_storage has length of seq_len + 1)
        self.x_storage = torch.empty(self.seq_capacity, seq_len + 1, x_size)
        self.u_storage = torch.empty(self.seq_capacity, seq_len, u_size)
        self.r_storage = torch.empty(self.seq_capacity, seq_len, 1)
        self.d_storage = torch.empty(self.seq_capacity, seq_len, 1,
                                     dtype=torch.bool)

        # Storage for hidden states (h associated with x, i associated with y)
        self.recurrent = r_size is not None
        if self.recurrent:
            self.r_size = r_size
            self.h0_storage = torch.zeros(self.seq_capacity, *r_size)
            self.hn_storage = torch.zeros(self.seq_capacity, *r_size)
        elif seq_len != 1:
            raise ValueError('Sequence length should be one without rnn')

        self.count = 0
        self.oldest = 0

    def add(self, x_seq, u_seq, r_seq, yn, d_seq, h0=None):
        """
        Add a new transition to the replay buffer.

        Arguments
        ---------
         - x_seq: tensor of shape (seq_len, x_size)
            Sequence of states
         - u_seq: tensor of shape (seq_len, u_size)
            Sequence of action taken
         - r_seq: tensor of shape (seq_len, 1)
            Sequence of rewards
         - y_seq: tensor of shape (seq_len, x_size)
            Sequence of next states
         - d_seq: tensor of shape (seq_len, 1)
            Sequence of dones
         - h0: None or tensor of shape `r_size`
            Hidden state of the rnn before the sequence
         - hn: None or tensor of shape `r_size`
            Hidden state of the rnn after the sequence
        """
        self.x_storage[self.oldest, :self.seq_len, :] = x_seq
        self.x_storage[self.oldest, -1, :] = yn
        self.u_storage[self.oldest, ...] = u_seq
        self.r_storage[self.oldest, ...] = r_seq
        self.d_storage[self.oldest, ...] = d_seq

        if self.recurrent:
            self.h0_storage[self.oldest, ...] = h0
            # self.hn_storage[self.oldest, ...] = hn

        self.oldest = (self.oldest + 1) % self.seq_capacity
        if self.count < self.seq_capacity:
            self.count += 1

    def sample(self, num_transitions, return_xu=False):
        """
        Sample transitions from the replay buffer.
        """
        if self.count == 0:
            raise ValueError('No transition has been stored in the buffer yet')

        random_indexes = torch.randint(self.count, (num_transitions,))

        XY_seq = self.x_storage[random_indexes, ...].transpose(0, 1)
        U_seq = self.u_storage[random_indexes, ...].transpose(0, 1)
        R_seq = self.r_storage[random_indexes, ...].transpose(0, 1)
        D_seq = self.d_storage[random_indexes, ...].transpose(0, 1)

        if self.recurrent:
            # The batch_size is the penultimate dimension by convention
            r_size = self.h0_storage.size()[1:]
            H_shape = r_size[:-1] + (num_transitions,) + r_size[-1:]
            H0 = self.h0_storage[random_indexes, ...].view(H_shape)
        else:
            H0 = None

        return XY_seq, U_seq, R_seq, D_seq, H0
