import torch
import torch.optim as optim
from quantum_emotion_net import QuantumEmotionNet

# Example setup
input_dim = 10      # number of features
hidden_dim = 16     # hidden layer size
output_dim = 3      # number of emotional states (e.g. happy, sad, neutral)

# Initialize model
model = QuantumEmotionNet(input_dim, hidden_dim, output_dim)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Dummy training data (replace with real dataset later)
X = torch.randn(100, input_dim)          # 100 samples, random features
y = torch.randint(0, output_dim, (100,)) # 100 labels (0,1,2)

# Training loop
for epoch in range(20):  # 20 epochs
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
