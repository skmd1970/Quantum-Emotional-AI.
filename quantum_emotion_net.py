import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumEmotionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QuantumEmotionNet, self).__init__()
        self.fc1_real = nn.Linear(input_dim, hidden_dim)
        self.fc1_imag = nn.Linear(input_dim, hidden_dim)
        self.fc2_real = nn.Linear(hidden_dim, output_dim)
        self.fc2_imag = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        real1 = self.fc1_real(x)
        imag1 = self.fc1_imag(x)
        z1 = torch.complex(real1, imag1)
        z1_mag = torch.abs(z1)
        real2 = self.fc2_real(z1_mag)
        imag2 = self.fc2_imag(z1_mag)
        z2 = torch.complex(real2, imag2)
        probs = torch.abs(z2) ** 2
        return F.softmax(probs, dim=1)
