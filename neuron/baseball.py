import math
import random

# 1. Activation Function (Sigmoid squashes numbers between 0 and 1)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 2. The Training Data (Inputs: [Temperature scaled 0-1, Rain 0=No/1=Yes])
# Target Output: 1 = Play Ball, 0 = Stay Home
# We want to play ONLY when it's warm (e.g., > 0.5) AND NOT raining (0)
training_data = [
    {"inputs": [0.2, 1.0], "target": 0},  # Freezing and Raining -> Don't Play
    {"inputs": [0.2, 0.0], "target": 0},  # Freezing and Dry     -> Don't Play
    {"inputs": [0.8, 1.0], "target": 0},  # Warm and Raining     -> Don't Play
    {"inputs": [0.9, 0.0], "target": 1},  # Perfect: Warm & Dry  -> PLAY BALL!
    {"inputs": [0.7, 0.0], "target": 1},  # Good: Warm & Dry     -> PLAY BALL!
]

# 3. Initialize Weights and Biases randomly
random.seed(42)  # For reproducible results

# Hidden Layer (2 Neurons, each taking 2 inputs)
w11, w12, b1 = random.uniform(-1, 1), random.uniform(-1, 1), 0.0
w21, w22, b2 = random.uniform(-1, 1), random.uniform(-1, 1), 0.0

# Output Layer (1 Neuron, taking 2 inputs from the hidden layer)
w_out1, w_out2, b_out = random.uniform(-1, 1), random.uniform(-1, 1), 0.0

learning_rate = 0.5

print("Training the 3-neuron umpire...")

# 4. The Training Loop (Epochs)
for epoch in range(5000):
    total_loss = 0

    for data in training_data:
        x1, x2 = data["inputs"]
        target = data["target"]

        # --- FORWARD PASS ---
        # Neuron 1 (Hidden)
        h1_input = (x1 * w11) + (x2 * w12) + b1
        h1_output = sigmoid(h1_input)

        # Neuron 2 (Hidden)
        h2_input = (x1 * w21) + (x2 * w22) + b2
        h2_output = sigmoid(h2_input)

        # Neuron 3 (Output)
        out_input = (h1_output * w_out1) + (h2_output * w_out2) + b_out
        final_output = sigmoid(out_input)

        # Calculate Error (Loss)
        error = target - final_output
        total_loss += error**2

        # --- BACKPROPAGATION (The Learning Part) ---
        # Output Layer Gradients
        d_output = error * sigmoid_derivative(final_output)

        # Hidden Layer Gradients
        d_h1 = d_output * w_out1 * sigmoid_derivative(h1_output)
        d_h2 = d_output * w_out2 * sigmoid_derivative(h2_output)

        # Update Weights and Biases for Output Neuron
        w_out1 += d_output * h1_output * learning_rate
        w_out2 += d_output * h2_output * learning_rate
        b_out += d_output * learning_rate

        # Update Weights and Biases for Hidden Neuron 1
        w11 += d_h1 * x1 * learning_rate
        w12 += d_h1 * x2 * learning_rate
        b1 += d_h1 * learning_rate

        # Update Weights and Biases for Hidden Neuron 2
        w21 += d_h2 * x1 * learning_rate
        w22 += d_h2 * x2 * learning_rate
        b2 += d_h2 * learning_rate

    # Print progress every 1000 steps
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Error: {total_loss:.4f}")

print("\nTraining Complete! Testing the Umpire:")
print("-" * 40)

# 5. Test the Trained Network on New Data
test_days = [
    {"desc": "Cold & Rainy day", "inputs": [0.1, 1.0]},
    {"desc": "Hot & Rainy day ", "inputs": [0.95, 1.0]},
    {"desc": "Cold & Sunny day ", "inputs": [0.15, 0.0]},
    {"desc": "Beautiful warm day", "inputs": [0.85, 0.0]},
]

for day in test_days:
    x1, x2 = day["inputs"]

    # Quick forward pass with our final weights
    h1 = sigmoid((x1 * w11) + (x2 * w12) + b1)
    h2 = sigmoid((x1 * w21) + (x2 * w22) + b2)
    prediction = sigmoid((h1 * w_out1) + (h2 * w_out2) + b_out)

    decision = "PLAY BALL!" if prediction > 0.5 else "STAY HOME"
    print(
        f"{day['desc']} -> Model Confidence: {prediction*100:5.1f}% -> Decision: {decision}"
    )