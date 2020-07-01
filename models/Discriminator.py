#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:09:57 2020

@author: diml
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(611, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
        
    def forward(self, seq_preds, hn):
        x = torch.cat([seq_preds, hn], 1).view(-1)
        out = self.model(x)
        
        return x
        
    