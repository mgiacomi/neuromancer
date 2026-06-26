import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Inputs and Target
x = 0.5
target = 0.8
learning_rate = 0.5

# Random Weights and Biases for two neurons
wA, bA = np.random.randn(), np.random.randn()
wB, bB = np.random.randn(), np.random.randn()

for epoch in range(1000):
    # --- FORWARD PASS ---
    z1 = x * wA + bA
    a1 = sigmoid(z1)  # Output of Neuron 1
    
    z2 = a1 * wB + bB
    a2 = sigmoid(z2)  # Final Output (Neuron 2)

    # --- BACKWARD PASS (Calculus) ---
    # 1. Error at the very end
    error = a2 - target
    
    # 2. Gradient for Neuron B (The "Easy" one)
    d_z2 = error * sigmoid_deriv(z2)
    d_wB = d_z2 * a1
    d_bB = d_z2
    
    # 3. Gradient for Neuron A (The "Chain" one)
    # We pass the error back through wB to Neuron A
    d_z1 = d_z2 * wB * sigmoid_deriv(z1)
    d_wA = d_z1 * x
    d_bA = d_z1

    # --- UPDATE WEIGHTS ---
    wB -= learning_rate * d_wB
    bB -= learning_rate * d_bB
    wA -= learning_rate * d_wA
    bA -= learning_rate * d_bA

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Output: {a2:.4f}")
