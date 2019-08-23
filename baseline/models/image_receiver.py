import torch
import torch.nn as nn
import numpy as np

import random

class ImageReceiver(nn.Module):

    def __init__(
        self,
        z_dim=11,
        hidden_dim=512,
        output_dim=30*30*3,
        embedding_dim=64):
        super(ImageReceiver, self).__init__()
        
        self.embedding_layer = nn.Embedding(z_dim, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())


    def forward(self, input):
        embedded_input = self.embedding_layer.forward(input)
        _, (lstm_output, _) = self.lstm_layer.forward(embedded_input)
        lstm_output = lstm_output.squeeze()
        output = self.output_layers.forward(lstm_output)
        return output
