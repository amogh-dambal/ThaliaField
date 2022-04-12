import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstmSize = 128
        self.embeddingDim = 128
        self.layers = 3

        N = len(dataset.uniqueWords)
        self.embedding = nn.Embedding(
            num_embeddings=N,
            embedding_dim=self.embeddingDim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstmSize,
            hidden_size=self.lstmSize,
            num_layers=self.layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstmSize, N)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def initState(self, seqLength):
        return (
            torch.zeros(self.layers, seqLength, self.lstmSize),
            torch.zeros(self.layers, seqLength, self.lstmSize),
        )
