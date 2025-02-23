# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_outputs, mask):
        # lstm_outputs: (batch, seq_len, hidden_dim)
        # mask: (batch, seq_len) with 1 for valid positions, 0 for padded
        attn_weights = self.attn(lstm_outputs).squeeze(-1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(2)  # (batch, seq_len, 1)
        weighted_output = torch.sum(lstm_outputs * attn_weights, dim=1)  # (batch, hidden_dim)
        return weighted_output

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_dim=258, hidden_dim=128, num_layers=2, num_classes=98,
                 bidirectional=True, use_attention=True, dropout=0.5):
        """
        input_dim: feature size (e.g. 33*4 + 21*3 + 21*3 = 258)
        """
        super(SignLanguageLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if use_attention:
            self.attention = Attention(lstm_output_dim)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
    # model.py (updated forward method)
    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        # lengths: (batch,) tensor of actual sequence lengths
        # Move lengths to CPU as required by pack_padded_sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Create mask based on lengths
        batch_size, seq_len, _ = outputs.size()
        device = outputs.device
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        if self.use_attention:
            out = self.attention(outputs, mask)
        else:
            # Use the last valid output per sequence
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, outputs.size(2))
            out = outputs.gather(1, idx).squeeze(1)
        out = self.fc(out)
        return out

