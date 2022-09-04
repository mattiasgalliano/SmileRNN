import torch
from torch import nn

# model
class SmileRNN(nn.Module):
    def __init__(self, n_unique_chars, char_embedding_dim, hidden_layer_dim, n_hidden_layers, device):
        super(SmileRNN, self).__init__()
        self.n_unique_chars = n_unique_chars
        self.char_embedding_dim = char_embedding_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.n_hidden_layers = n_hidden_layers
        self.device = device

        self.char_embedding = nn.Embedding(
            num_embeddings=n_unique_chars + 1, # EOS
            embedding_dim=char_embedding_dim
        ).to(device)

        self.rnn = nn.GRU(
            input_size=char_embedding_dim,
            hidden_size=hidden_layer_dim,
            num_layers=n_hidden_layers,
            batch_first=True
        ).to(device)

        self.output = nn.Sequential(
            nn.Linear(hidden_layer_dim, n_unique_chars + 1), # EOS
            # nn.Dropout(0.1),
            nn.LogSoftmax(dim=2)
        ).to(device)

    def init_hidden(self):
        empty = torch.empty(self.n_hidden_layers, 1, self.hidden_layer_dim).to(self.device)
        h = torch.nn.init.xavier_normal_(empty) # TODO change init?
        return h
        
    def forward(self, x, h):
        embedded_x = self.char_embedding(x)
        embedded_x = embedded_x.unsqueeze(0)
        o, h = self.rnn(embedded_x, h)
        out = self.output(o)
        return out, h

    def embed(self, x):
        embedded_x = self.char_embedding(x)
        embedded_x = embedded_x.unsqueeze(0)
        return x
