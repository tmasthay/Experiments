import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchsummary import summary
import os
import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch Wine Training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

# Generate random data
n_train_samples, n_test_samples = 10000, 100
n_features = 50

train_features = torch.randn(n_train_samples, n_features).to(device)
train_targets = torch.randn(n_train_samples, 1).to(device)
test_features = torch.randn(n_test_samples, n_features).to(device)
test_targets = torch.randn(n_test_samples, 1).to(device)

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'B'

size, unit = format_bytes(n_train_samples * n_features * 8)
print(f'Total size of training data: {size:.2f} {unit}')

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(n_features, 64), 
            nn.Linear(64, 500),
            nn.Linear(500, 1000),
            nn.Linear(1000,1000),
            nn.Linear(1000, 1)
        ])
        identity = lambda x : x 
        self.activations = [torch.relu, torch.relu, torch.relu, torch.relu, identity]
        
    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x

model = Net().to(device)
summary(model, input_size=(n_features,))
os.system('sleep 5')

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 500
start_time = time.time()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_targets)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
end_time = time.time()

print('Training complete')
print('%d epochs took %f seconds.'%(epochs, end_time - start_time))

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(test_features)
    test_loss = criterion(predictions, test_targets)
print('Test Loss: {:.4f}'.format(test_loss.item()))

