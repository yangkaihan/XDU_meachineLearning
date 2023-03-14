import torch
import torch.nn as nn
import torch.optim as optim


# Define the MLP architecture
class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.hidden_layer = nn.Linear(3, 4)
        self.output_layer = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden_layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x


# Create an instance of the MLP
mlp = XOR_MLP()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.1)

# Define the training data
X = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                 dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=torch.float32)

# Train the MLP
for epoch in range(5000):
    # Forward pass
    outputs = mlp(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 5000, loss.item()))

# Print the MLP parameters
print('Hidden layer weight:', mlp.hidden_layer.weight)
print('Hidden layer bias:', mlp.hidden_layer.bias)
print('Output layer weight:', mlp.output_layer.weight)
print('Output layer bias:', mlp.output_layer.bias)

# Test the MLP
with torch.no_grad():
    test_input = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                              dtype=torch.float32)
    test_output = mlp(test_input)
    print('Test output:')
    print(test_output)