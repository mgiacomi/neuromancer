import os
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Device selection: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load and parse the dataset
# Locate breast_cancer_data.csv in the same directory as the script
data_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(data_dir, "breast_cancer_data.csv")

raw_features = []
raw_targets = []
with open(data_path, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header
    for row in reader:
        if not row:
            continue
        features = [float(val) for val in row[:-1]]
        target = int(float(row[-1]))
        raw_features.append(features)
        raw_targets.append(target)

# Convert to PyTorch tensors and send to device
X_all = torch.tensor(raw_features, dtype=torch.float32, device=device)
y_all = torch.tensor(raw_targets, dtype=torch.float32, device=device).view(-1, 1)

# 3. Train-Test Split (75% Train, 25% Test) - Random split on every run
num_samples = len(X_all)
indices = torch.randperm(num_samples, device=device)
split_idx = int(num_samples * 0.75)

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X_all[train_indices]
y_train = y_all[train_indices]
X_test = X_all[test_indices]
y_test = y_all[test_indices]

# 4. Feature Scaling (Z-Score Standardization based on Training Data)
means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True, correction=0)
stds[stds == 0.0] = 1.0

X_train_scaled = (X_train - means) / stds
X_test_scaled = (X_test - means) / stds

# 5. Define Neural Network Model
class CancerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Apply manual Xavier initialization (matches Cain's setup)
        limit1 = math.sqrt(6.0 / (input_dim + hidden_dim))
        nn.init.uniform_(self.fc1.weight, -limit1, limit1)
        nn.init.zeros_(self.fc1.bias)
        
        limit2 = math.sqrt(6.0 / (hidden_dim + 1))
        nn.init.uniform_(self.fc2.weight, -limit2, limit2)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h1 = torch.sigmoid(self.fc1(x))
        # Return raw logits for numerical stability with BCEWithLogitsLoss
        return self.fc2(h1)

num_features = X_train.shape[1]
num_hidden = 16
model = CancerClassifier(num_features, num_hidden).to(device)

# 6. Setup DataLoader for vectorized mini-batching
batch_size = 32
train_dataset = TensorDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Hyperparameters
initial_lr = 0.1
decay_rate = 0.002
beta = 0.9      # Momentum coefficient
l2_reg = 0.001  # L2 Regularization (Weight Decay)
epochs = 1000

# Setup Optimizer and Loss Function
optimizer = optim.SGD(
    model.parameters(),
    lr=initial_lr,
    momentum=beta,
    nesterov=True,
    weight_decay=l2_reg
)
criterion = nn.BCEWithLogitsLoss()

print(f"Loaded {num_samples} samples.")
print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples.")
print(f"Training the 1-hidden-layer neural network with {num_hidden} neurons...")

# 7. Training Loop (using vectorized mini-batches and PyTorch native SGD)
for epoch in range(epochs + 1):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    
    # Apply learning rate decay
    lr = initial_lr / (1.0 + decay_rate * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        # Accumulate loss on device without CPU-GPU sync
        total_loss += loss.detach() * batch_x.size(0)
        
    if epoch % 100 == 0:
        avg_loss = (total_loss / len(X_train)).item()
        print(f"Epoch {epoch:4d} | lr: {lr:.5f} | Average Train BCE Loss: {avg_loss:.5f}")

# 8. Evaluate on Test Data
correct = 0
print("\nEvaluating on Test Data (remaining 25%):")
print("-" * 65)
print(f"{'Sample #':<10} | {'True Target':<12} | {'Predicted Prob':<16} | {'Decision':<10} | {'Correct?':<8}")
print("-" * 65)

# Calculate all predictions at once on the GPU/CPU for efficiency
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(X_test_scaled)).view(-1)
    
targets = y_test.view(-1)
decisions = (predictions > 0.5).long()
corrects = (decisions == targets.long())
correct_count = corrects.sum().item()

for idx in range(len(X_test)):
    target_val = int(targets[idx].item())
    pred_val = predictions[idx].item()
    decision_val = int(decisions[idx].item())
    is_correct = bool(corrects[idx].item())
    
    if idx < 20:  # Print first 20 for brief display
        print(f"{idx+1:<10} | {target_val:<12} | {pred_val:>14.2%} | {decision_val:<10} | {'YES' if is_correct else 'NO'}")

if len(X_test) > 20:
    print(f"... ({len(X_test) - 20} more test cases hidden)")

accuracy = (correct_count / len(X_test)) * 100
print("-" * 65)
print(f"Final Test Accuracy: {correct_count}/{len(X_test)} ({accuracy:.2f}%)")
