import torch
import torch.nn as nn

from .darts_cell import DARTSCell
from .shapes_meta_visual_module import ShapesMetaVisualModule

class ShapesReceiver(nn.Module):
    def __init__(
        self,
        vocab_size,
        device,
        embedding_size=256,
        hidden_size=512):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.device = device

        self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )
        # self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)
        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
            )

    def forward(self, messages=None):
        batch_size = messages.shape[0]

        emb = torch.matmul(messages, self.embedding) if self.training else self.embedding[messages]
        # emb = self.embedding.forward(messages)

        # initialize hidden
        h = torch.zeros([batch_size, self.hidden_size], device=self.device)
        c = torch.zeros([batch_size, self.hidden_size], device=self.device)
        h = (h, c)

        # make sequence_length be first dim
        seq_iterator = emb.transpose(0, 1)
        for w in seq_iterator:
            h = self.rnn(w, h)

        h = h[0]  # keep only hidden state

        out = h

        return out, emb
