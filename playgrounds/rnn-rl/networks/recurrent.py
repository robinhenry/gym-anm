# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from utils.gym import check_u_size, action_scaling


class RNN(nn.RNN):
    def __init__(self, input_size, hidden_size, num_layers):
        self.r_size = (num_layers, hidden_size)
        super().__init__(input_size, hidden_size, num_layers)


class GRU(nn.GRU):
    def __init__(self, input_size, hidden_size, num_layers):
        self.r_size = (num_layers, hidden_size)
        super().__init__(input_size, hidden_size, num_layers)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.r_size = (2, num_layers, hidden_size)

    def forward(self, x, h):
        h, c = h[0, ...], h[1, ...]
        output, (hn, cn) = self.lstm(x, (h, c))
        return output, torch.stack((hn, cn))


class BRCLayer(nn.Module):
    """
    Recurrent Neural Network (single layer) using the Bistable Recurrent Cell
    (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        self.U_c = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, input_size)))
        self.w_c = nn.Parameter(nn.init.normal_(torch.Tensor(hidden_size)))
        self.b_c = nn.Parameter(nn.init.normal_(
            torch.Tensor(hidden_size)))

        # Reset gate (bistability feedback parameter)
        self.U_a = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, input_size)))
        self.w_a = nn.Parameter(nn.init.normal_(torch.Tensor(hidden_size)))
        self.b_a = nn.Parameter(nn.init.normal_(
            torch.Tensor(hidden_size)))

        # Hidden state
        self.U_h = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, input_size)))

    def forward(self, x_seq, h):
        """
        Compute the forward pass for the whole sequence.

        Arguments
        ---------
        - x_seq: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h: tensor of shape (batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        # Forward pass along all sequence
        output = torch.empty(seq_len, batch_size, self.hidden_size).to(x_seq)

        for t, x in enumerate(x_seq):
            c = torch.sigmoid(torch.matmul(x, self.U_c.T) +
                              self.w_c * h + self.b_c)
            a = 1 + torch.tanh(torch.matmul(x, self.U_a.T) +
                               self.w_a * h + self.b_a)
            h = c * h + (1 - c) * \
                torch.tanh(torch.matmul(x, self.U_h.T) + a * h)
            output[t, ...] = h

        return output, h


class NBRCLayer(nn.Module):
    """
    Recurrent Neural Network (single layer) using the Recurrently
    Neuromodulated Bistable Recurrent Cell (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        self.U_c = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, input_size)))
        self.W_c = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, hidden_size)))
        self.b_c = nn.Parameter(nn.init.normal_(
            torch.Tensor(hidden_size)))

        # Reset gate (bistability feedback parameter)
        self.U_a = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, input_size)))
        self.W_a = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, hidden_size)))
        self.b_a = nn.Parameter(nn.init.normal_(
            torch.Tensor(hidden_size)))

        # Hidden state
        self.U_h = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(hidden_size, input_size)))

    def forward(self, x_seq, h):
        """
        Compute the forward pass for the whole sequence.

        Arguments
        ---------
        - x_seq: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h: tensor of shape (batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        # Forward pass along all sequence
        output = torch.empty(seq_len, batch_size, self.hidden_size).to(x_seq)

        for t, x in enumerate(x_seq):
            c = torch.sigmoid(torch.matmul(x, self.U_c.T) +
                              torch.matmul(h, self.W_c.T) + self.b_c)
            a = 1 + torch.tanh(torch.matmul(x, self.U_a.T) +
                               torch.matmul(h, self.W_a.T) + self.b_a)
            h = c * h + (1 - c) * \
                torch.tanh(torch.matmul(x, self.U_h.T) + a * h)
            output[t, ...] = h

        return output, h


class BRC(nn.Module):
    """
    Recurrent Neural Network using the [Recurrently Neuromodulated] Bistable
    Recurrent Cell (see arXiv:2006.05252), with several stacked [n]BRC.
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 neuromodulated=False):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size
        - num_layers: int
            Number of stacked RNNs
        - neuromodulated: bool
            Whether to use neuromodulation (i.e. NBRCLayer instead of BRCLayer)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r_size = (num_layers, hidden_size)

        # Initialize the parameters for each layer of stacked RNN
        self.layers = []
        for layer in range(num_layers):
            if neuromodulated:
                stack = NBRCLayer(input_size, hidden_size)
            else:
                stack = BRCLayer(input_size, hidden_size)
            self.layers.append(stack)

            # For the non first layers, input is the output of the previous one
            input_size = hidden_size
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, h):
        """
        Compute the forward pass for the whole sequence and along each layers.

        Arguments
        ---------
        - x: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h: tensor of shape (num_layers, batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (num_layers, batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        if x.dim() != 3 or x.size(2) != self.input_size:
            raise ValueError('x should be of shape (seq_len, batch_size, '
                             'input_size')
        if (h.dim() != 3 or h.size(0) != len(self.layers) or
                h.size(2) != self.hidden_size):
            raise ValueError('h should of shape (num_layers, batch_size, '
                             'hidden_size')
        if h.size(1) != x.size(1):
            raise ValueError('The batch size of x and h do not correspond')

        seq_len = x.size(0)
        batch_size = x.size(1)

        hn = torch.empty(len(self.layers), batch_size, self.hidden_size).to(x)
        for i, (layer, h0) in enumerate(zip(self.layers, h)):
            x, h = layer(x, h0)
            hn[i, ...] = h

        return x, hn


class NBRC(BRC):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(input_size, hidden_size, num_layers,
                         neuromodulated=True)
