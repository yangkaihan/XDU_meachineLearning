import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP class
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    # Define the layers
    self.input_layer = nn.Linear(2, 4)
    self.hidden_layer = nn.Linear(4, 1)
    # Define the activation function
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # Pass the input through the layers and activation function
    x = self.input_layer(x)
    x = self.sigmoid(x)
    x = self.hidden_layer(x)
    x = self.sigmoid(x)
    return x

# Create an instance of the MLP class
mlp = MLP()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.01)

# Define the input and output data for XOR gate
x_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y_data = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

# Train the MLP for 1000 epochs
for epoch in range(10000):
  # Forward pass
  y_pred = mlp(x_data)
  # Compute the loss
  loss = criterion(y_pred, y_data)
  # Print the loss every 100 epochs

  # Zero the gradients
  optimizer.zero_grad()
  # Backward pass
  loss.backward()
  # Update the parameters
  optimizer.step()

# Test the MLP on XOR gate
y_test = mlp(x_data)
print(y_test)
y_test[y_test>0.5]=1
y_test[y_test<0.5]=0
print(f"Output: {y_test}")
