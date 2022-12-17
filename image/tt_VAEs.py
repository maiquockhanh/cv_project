# uncompyle6 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.7.10 (default, Dec 13 2022, 23:54:53) 
# [Clang 13.1.6 (clang-1316.0.21.2.5)]
# Embedded file name: /home/ma-prof/xma-group/huck/0_Speech/Tensor-Train-Neural-Network/image/tt_VAEs.py
# Compiled at: 2019-11-20 03:56:17
# Size of source mod 2**32: 2749 bytes
import os, sys, numpy as np, argparse, time, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tc.tc_fc import TTLinear

class tt_autoencoder(nn.Module):

    def __init__(self, hidden_tensors, input_tensor, output_dim, tt_rank):
        super(tt_autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            TTLinear(input_tensor, (hidden_tensors[0]), tt_rank=tt_rank), 
            TTLinear((hidden_tensors[0]), (hidden_tensors[1]), tt_rank=tt_rank), 
            TTLinear((hidden_tensors[1]), (hidden_tensors[2]), tt_rank=tt_rank)
        )
        self.decoder1 = nn.Sequential(
            TTLinear((hidden_tensors[2]), (hidden_tensors[1]), tt_rank=tt_rank), 
            TTLinear((hidden_tensors[1]), (hidden_tensors[0]), tt_rank=tt_rank), 
            TTLinear((hidden_tensors[0]), input_tensor, tt_rank=tt_rank)
        )
        self.lin = nn.Linear(
            np.prod(input_tensor), np.prod(input_tensor)
        )
        
        self.model_name = 'Tensor_Train_Autoencoder'
        
        

    def forward(self):
        out = self.encoder1(inputs)
        out = torch.sigmoid(self.lin(self.decoder1(out)))
        return out


class tt_VAE(nn.Module):

    def __init__(self, hidden_tensors, input_tensor, output_dim, tt_rank):
        super(tt_VAE, self).__init__()
        self.encoder1 = nn.Sequential(TTLinear(input_tensor, (hidden_tensors[0]), tt_rank=tt_rank), TTLinear((hidden_tensors[0]), (hidden_tensors[1]), tt_rank=tt_rank))
        self.fc21 = TTLinear((hidden_tensors[1]), (hidden_tensors[2]), tt_rank=tt_rank)
        self.fc22 = TTLinear((hidden_tensors[1]), (hidden_tensors[2]), tt_rank=tt_rank)
        self.decoder1 = nn.Sequential(TTLinear((hidden_tensors[2]), (hidden_tensors[1]), tt_rank=tt_rank), TTLinear((hidden_tensors[1]), (hidden_tensors[0]), tt_rank=tt_rank), TTLinear((hidden_tensors[0]), input_tensor, tt_rank=tt_rank))
        self.lin = nn.Linear(np.prod(input_tensor), np.prod(input_tensor))

    def encoder(self, x):
        out = self.encoder1(x)
        return (self.fc21(out), self.fc32(out))

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        out = F.sigmoid(self.lin(self.decoder1(z)))
        return out

    def forward(self, x):
        mu, log_var = self.encoder1(x)
        z = self.sampling(mu, log_var)
        return (self.decoder(z), mu, log_var)
# okay decompiling tt_VAEs.cpython-37.pyc
