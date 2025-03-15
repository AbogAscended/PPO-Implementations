import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_channels, gru_hidden_size, num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, 256),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden_size,
        )
        
        self.policy = nn.Linear(gru_hidden_size, num_actions)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        cnn_out = self.cnn(x.view(batch_size, *x.shape[2:]))
        cnn_out = cnn_out.unsqueeze(0)
        
        gru_out, new_hidden = self.gru(cnn_out, hidden)
        
        logits = self.policy(gru_out.squeeze(0))
        return logits, new_hidden