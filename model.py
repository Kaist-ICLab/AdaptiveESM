import torch.nn as nn


class LSTM(nn.Module):
    """Baseline LSTM class"""
    def __init__(self, input_size, output_size, hidden_size, num_layers, p_drop=0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=p_drop)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.contiguous()
        out = self.fc(out)
        out = out[:, -1]

        return out
