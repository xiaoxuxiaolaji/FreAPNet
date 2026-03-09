import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MultiheadAttention

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embed_size = 128 #embed_size
        self.hidden_size = 256 #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        
        # Phase-Amplitude parameterization
        self.amp1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.phase1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.amp_bias1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.phase_bias1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        
        self.amp2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.phase2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.amp_bias2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.phase_bias2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        
        # Adaptive frequency masks with sigmoid gate
        self.freq_mask1 = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.freq_mask2 = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.mask_gate = nn.Sigmoid()

        # Enhanced attention with frequency-aware bias
        self.attention = MultiheadAttention(
            embed_dim=self.embed_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        # Frequency-aware attention bias
        self.freq_bias = nn.Parameter(torch.zeros(1, self.embed_size))
        
        # Enhanced FC layers with residual connections
        self.fc1 = nn.Linear(self.seq_length * self.embed_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.pre_length)
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.amp2, self.phase2, self.amp_bias2, self.phase_bias2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.amp1, self.phase1, self.amp_bias1, self.phase_bias1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, amp, phase, amp_bias, phase_bias):
        # Ensure device consistency
        device = x.device
        amp = amp.to(device)
        phase = phase.to(device)
        amp_bias = amp_bias.to(device)
        phase_bias = phase_bias.to(device)
        
        # Apply gated frequency mask
        if amp is self.amp1:
            mask = self.mask_gate(self.freq_mask1.to(device))
        else:
            mask = self.mask_gate(self.freq_mask2.to(device))
        x = x * mask
        
        # Convert to polar form with safe operations
        try:
            mag = torch.abs(x)
            ang = torch.angle(x)
            
            # Phase-Amplitude processing
            new_mag = F.relu(
                torch.einsum('bijd,dd->bijd', mag, amp) + \
                amp_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            
            new_ang = F.relu(
                torch.einsum('bijd,dd->bijd', ang, phase) + \
                phase_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            
            # Convert back to complex and apply softshrink to real/imag separately
            y_real = new_mag * torch.cos(new_ang)
            y_imag = new_mag * torch.sin(new_ang)
            y_real = F.softshrink(y_real, lambd=self.sparsity_threshold)
            y_imag = F.softshrink(y_imag, lambd=self.sparsity_threshold)
            return torch.complex(y_real, y_imag)
            
        except RuntimeError as e:
            # Fallback to CPU if CUDA fails
            x_cpu = x.cpu()
            mag = torch.abs(x_cpu)
            ang = torch.angle(x_cpu)
            new_mag = F.relu(
                torch.einsum('bijd,dd->bijd', mag, amp.cpu()) + \
                amp_bias.cpu().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            new_ang = F.relu(
                torch.einsum('bijd,dd->bijd', ang, phase.cpu()) + \
                phase_bias.cpu().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            # Apply softshrink to real/imag separately
            y_real = new_mag * torch.cos(new_ang)
            y_imag = new_mag * torch.sin(new_ang)
            y_real = F.softshrink(y_real, lambd=self.sparsity_threshold)
            y_imag = F.softshrink(y_imag, lambd=self.sparsity_threshold)
            return torch.complex(y_real, y_imag).to(device)

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        
        # Enhanced attention with frequency bias
        x_attn = x.reshape(B*N, T, self.embed_size)
        # Add frequency-aware bias
        attn_bias = self.freq_bias * torch.linspace(0, 1, T, device=x.device).view(1, T, 1)
        x_attn = x_attn + attn_bias
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)
        x_attn = attn_output.reshape(B, N, T, self.embed_size)
        
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x_attn, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        
        # Enhanced FC processing with residual connections
        x = x.reshape(B, N, -1)
        residual = self.fc1(x)
        x = F.leaky_relu(self.ln1(residual))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.leaky_relu(self.ln2(x + residual))
        x = self.dropout(x)
        x = self.fc3(x).permute(0, 2, 1)
        
        return x
