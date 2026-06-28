import csv
import random
import math

# 1. Activation Functions
def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Load and parse the dataset
data_path = "/Users/caing/code/neuromancer/neuron/breast_cancer_data.csv"

raw_data = []
with open(data_path, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header
    for row in reader:
        if not row:
            continue
        features = [float(val) for val in row[:-1]]
        target = int(float(row[-1]))
        raw_data.append({"features": features, "target": target})

# 3. Train-Test Split (25% Train, 75% Test) - Random split on every run
random.shuffle(raw_data)

split_idx = int(len(raw_data) * 0.75)
train_data = raw_data[:split_idx]
test_data = raw_data[split_idx:]

# 4. Feature Scaling (Z-Score Standardization based on Training Data)
num_features = len(train_data[0]["features"])
means = [0.0] * num_features
stds = [0.0] * num_features
n_train = len(train_data)

# Compute means
for item in train_data:
    for j in range(num_features):
        means[j] += item["features"][j]
for j in range(num_features):
    means[j] /= n_train

# Compute standard deviations
for item in train_data:
    for j in range(num_features):
        stds[j] += (item["features"][j] - means[j]) ** 2
for j in range(num_features):
    stds[j] = math.sqrt(stds[j] / n_train)

# Scale function using Z-score standardization
def scale_features(features):
    scaled = []
    for j in range(num_features):
        if stds[j] == 0:
            scaled.append(0.0)
        else:
            scaled.append((features[j] - means[j]) / stds[j])
    return scaled

# Apply scaling
for item in train_data:
    item["scaled_features"] = scale_features(item["features"])

for item in test_data:
    item["scaled_features"] = scale_features(item["features"])

# 5. Xavier Initialization
# Hidden Layer: 16 Neurons, fan_in = 30, fan_out = 16
num_hidden = 16
limit1 = math.sqrt(6.0 / (num_features + num_hidden))
w1 = [[random.uniform(-limit1, limit1) for _ in range(num_features)] for _ in range(num_hidden)]
b1 = [0.0] * num_hidden

# Output Layer: 1 Neuron, fan_in = 16, fan_out = 1
limit2 = math.sqrt(6.0 / (num_hidden + 1))
w2 = [random.uniform(-limit2, limit2) for _ in range(num_hidden)]
b2 = 0.0

# 6. Initialize Momentum Velocities
v_w1 = [[0.0] * num_features for _ in range(num_hidden)]
v_b1 = [0.0] * num_hidden
v_w2 = [0.0] * num_hidden
v_b2 = 0.0

# Hyperparameters
initial_lr = 0.8
decay_rate = 0.0005
beta = 0.7      # Momentum coefficient
l2_reg = 0.001  # L2 Regularization (Weight Decay)
epochs = 500

print(f"Loaded {len(raw_data)} samples.")
print(f"Training set: {len(train_data)} samples, Test set: {len(test_data)} samples.")
print(f"Training the 1-hidden-layer neural network with {num_hidden} neurons...")

# 7. Training Loop (using Nesterov Accelerated Gradient)
for epoch in range(epochs + 1):
    total_loss = 0.0
    
    # Apply learning rate decay
    lr = initial_lr / (1.0 + decay_rate * epoch)
    
    # Shuffle training data each epoch
    random.shuffle(train_data)
    
    for item in train_data:
        x = item["scaled_features"]
        target = item["target"]
        
        # Nesterov lookahead weights
        w1_la = [[w1[i][j] + beta * v_w1[i][j] for j in range(num_features)] for i in range(num_hidden)]
        b1_la = [b1[i] + beta * v_b1[i] for i in range(num_hidden)]
        w2_la = [w2[i] + beta * v_w2[i] for i in range(num_hidden)]
        b2_la = b2 + beta * v_b2
        
        # Forward pass
        h1 = [sigmoid(sum(x[j] * w1_la[i][j] for j in range(num_features)) + b1_la[i]) for i in range(num_hidden)]
        out_input = sum(h1[i] * w2_la[i] for i in range(num_hidden)) + b2_la
        prediction = sigmoid(out_input)
        
        # BCE Loss calculation
        eps = 1e-15
        pred_clipped = max(eps, min(1 - eps, prediction))
        loss = -(target * math.log(pred_clipped) + (1 - target) * math.log(1 - pred_clipped))
        total_loss += loss
        
        # Gradients
        # For output node: dL/d(out_input) = prediction - target
        d_out = prediction - target
        
        # Backprop through output weight to H1
        d_h1 = [d_out * w2_la[i] * sigmoid_derivative(h1[i]) for i in range(num_hidden)]
        
        # Gradient values with L2 regularization derivatives added
        g_w2 = [d_out * h1[i] + l2_reg * w2[i] for i in range(num_hidden)]
        g_b2 = d_out
        
        g_w1 = [[d_h1[i] * x[j] + l2_reg * w1[i][j] for j in range(num_features)] for i in range(num_hidden)]
        g_b1 = [d_h1[i] for i in range(num_hidden)]
        
        # Update velocities and weights (Momentum + Gradient Descent)
        v_b2 = beta * v_b2 - lr * g_b2
        b2 += v_b2
        
        for i in range(num_hidden):
            v_w2[i] = beta * v_w2[i] - lr * g_w2[i]
            w2[i] += v_w2[i]
            
            v_b1[i] = beta * v_b1[i] - lr * g_b1[i]
            b1[i] += v_b1[i]
            for j in range(num_features):
                v_w1[i][j] = beta * v_w1[i][j] - lr * g_w1[i][j]
                w1[i][j] += v_w1[i][j]
        
    if epoch % 100 == 0:
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch:4d} | lr: {lr:.5f} | Average Train BCE Loss: {avg_loss:.5f}")

# 8. Evaluate on Test Data
correct = 0
print("\nEvaluating on Test Data (remaining 25%):")
print("-" * 65)
print(f"{'Sample #':<10} | {'True Target':<12} | {'Predicted Prob':<16} | {'Decision':<10} | {'Correct?':<8}")
print("-" * 65)

for idx, item in enumerate(test_data):
    x = item["scaled_features"]
    target = item["target"]
    
    # Forward pass
    h1 = [sigmoid(sum(x[j] * w1[i][j] for j in range(num_features)) + b1[i]) for i in range(num_hidden)]
    prediction = sigmoid(sum(h1[i] * w2[i] for i in range(num_hidden)) + b2)
    
    decision = 1 if prediction > 0.5 else 0
    is_correct = decision == target
    if is_correct:
        correct += 1
    
    if idx < 20:  # Print first 20 for brief display
        print(f"{idx+1:<10} | {target:<12} | {prediction:>14.2%} | {decision:<10} | {'YES' if is_correct else 'NO'}")

if len(test_data) > 20:
    print(f"... ({len(test_data) - 20} more test cases hidden)")

accuracy = (correct / len(test_data)) * 100
print("-" * 65)
print(f"Final Test Accuracy: {correct}/{len(test_data)} ({accuracy:.2f}%)")
