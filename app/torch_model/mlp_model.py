from torch import nn
import torch.nn.functional as F


class MLP_Model(nn.Module):
    def __init__(self, features: int = 100, hidden: int = 64, n_output: int = 1):
        super().__init__()
        self.name = "MLP"
        self.features = features
        self.hidden = hidden
        self.n_output = n_output

        self.linear1 = nn.Linear(self.features, self.hidden)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(self.hidden, self.n_output)

    def forward(self, x):
        out = F.relu(self.dropout(self.linear1(x)))
        out = self.linear2(out)
        return out
