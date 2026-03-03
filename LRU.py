"""
Linear Recurrent Unit (LRU) Implementation
Based on "Resurrecting Recurrent Neural Networks for Long Sequences"

The LRU uses a diagonal linear recurrence with complex-valued parameters
for efficient and stable long-sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class LRU(nn.Module):
    """
    Linear Recurrent Unit (LRU)

    Implements a diagonal linear recurrence:
        h_t = A * h_{t-1} + B * x_t
        y_t = Re(C * h_t + D * x_t)

    where A is a diagonal complex matrix, and the computation is done in complex domain
    for better long-term dependency modeling.

    Arguments:
        d_model: Model dimension (input and output dimension)
        d_hidden: Hidden state dimension (recurrent state size)
        r_min: Minimum value for eigenvalue magnitudes
        r_max: Maximum value for eigenvalue magnitudes
        max_phase: Maximum phase for eigenvalues (in multiples of pi)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.283,  # 2*pi
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        # Initialize diagonal A matrix parameters (complex-valued)
        # A is parameterized as: A = exp(-exp(nu) + i*theta)
        # This ensures |A| < 1 for stability
        self.nu_log = nn.Parameter(torch.randn(d_hidden))
        self.theta_log = nn.Parameter(torch.randn(d_hidden))

        # B matrix: projects input to hidden state (complex-valued)
        self.B_re = nn.Parameter(torch.randn(d_hidden, d_model) / np.sqrt(2 * d_model))
        self.B_im = nn.Parameter(torch.randn(d_hidden, d_model) / np.sqrt(2 * d_model))

        # C matrix: projects hidden state to output (complex-valued)
        self.C_re = nn.Parameter(torch.randn(d_model, d_hidden) / np.sqrt(d_hidden))
        self.C_im = nn.Parameter(torch.randn(d_model, d_hidden) / np.sqrt(d_hidden))

        # D matrix: skip connection (real-valued)
        self.D = nn.Parameter(torch.randn(d_model))

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def _compute_lambda(self):
        """
        Compute the diagonal elements of A matrix.
        Lambda = exp(-exp(nu) + i*theta)
        This ensures |lambda| < 1 for stability.
        """
        # Compute magnitude: r = exp(-exp(nu))
        # This ensures 0 < r < 1
        nu = torch.tanh(self.nu_log)
        r = torch.exp(-torch.exp(nu * (np.log(self.r_max) - np.log(self.r_min)) + np.log(self.r_min)))

        # Compute phase: theta
        theta = torch.tanh(self.theta_log) * self.max_phase

        # Compute complex eigenvalues
        lambda_re = r * torch.cos(theta)
        lambda_im = r * torch.sin(theta)

        return lambda_re, lambda_im

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Forward pass of LRU.

        Arguments:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional initial hidden state (batch_size, d_hidden, 2) where last dim is [real, imag]
                   If None, initialize with zeros.

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            final_state: Final hidden state (batch_size, d_hidden, 2)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get diagonal elements of A
        lambda_re, lambda_im = self._compute_lambda()

        # Convert B and C to complex
        B_complex = torch.complex(self.B_re, self.B_im)  # (d_hidden, d_model)
        C_complex = torch.complex(self.C_re, self.C_im)  # (d_model, d_hidden)

        # Initialize hidden state if not provided
        if state is None:
            h_re = torch.zeros(batch_size, self.d_hidden, device=device)
            h_im = torch.zeros(batch_size, self.d_hidden, device=device)
        else:
            h_re = state[:, :, 0]
            h_im = state[:, :, 1]

        outputs = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, d_model)

            # Compute B * x_t in complex domain
            Bx = torch.matmul(x_t, B_complex.t())  # (batch_size, d_hidden) complex
            Bx_re = Bx.real
            Bx_im = Bx.imag

            # Update hidden state: h_t = lambda * h_{t-1} + B * x_t
            # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            h_re_new = lambda_re * h_re - lambda_im * h_im + Bx_re
            h_im_new = lambda_re * h_im + lambda_im * h_re + Bx_im

            h_re = h_re_new
            h_im = h_im_new

            # Compute output: y_t = Re(C * h_t) + D * x_t
            h_complex = torch.complex(h_re, h_im)
            Ch = torch.matmul(h_complex, C_complex.t())  # (batch_size, d_model) complex

            y_t = Ch.real + self.D * x_t  # Take real part and add skip connection
            outputs.append(y_t)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
        output = self.norm(output)

        # Prepare final state
        final_state = torch.stack([h_re, h_im], dim=-1)  # (batch_size, d_hidden, 2)

        return output, final_state


class LRUBlock(nn.Module):
    """
    LRU Block with pre-normalization and residual connection.

    This wraps an LRU layer with:
    - Layer normalization
    - Residual connection
    - Optional feedforward network

    Arguments:
        d_model: Model dimension
        d_hidden: Hidden state dimension for LRU
        ffn_dim: Feedforward network dimension (if None, no FFN)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.lru = LRU(d_model, d_hidden)
        self.dropout1 = nn.Dropout(dropout)

        # Optional feedforward network
        self.ffn = None
        if ffn_dim is not None:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Forward pass with residual connections.

        Arguments:
            x: Input tensor (batch_size, seq_len, d_model)
            state: Optional initial hidden state

        Returns:
            output: Output tensor (batch_size, seq_len, d_model)
            final_state: Final hidden state
        """
        # LRU with residual
        normed_x = self.norm1(x)
        lru_out, final_state = self.lru(normed_x, state)
        x = x + self.dropout1(lru_out)

        # Optional FFN with residual
        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))

        return x, final_state
