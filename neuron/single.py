import numpy as np

# Inputs (e.g., hours studied, coffee consumed)
x = np.array([2.0, 3.0]) 
# Initial random weights and bias
w = np.array([0.5, -0.2])
b = 0.1

# The Linear Algebra step
z = np.dot(x, w) + b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

output = sigmoid(z)
print(f"Neuron Output: {output}")

learning_rate = 0.1
target = 1.0  # We want the neuron to eventually output 1

for epoch in range(100):
    # 1. Forward Pass
    z = np.dot(x, w) + b
    output = sigmoid(z)
    
    # 2. Calculate Error
    error = output - target
    
    # 3. Backpropagation (The Calculus)
    # Derivative of sigmoid is: sigmoid(z) * (1 - sigmoid(z))
    d_output_dz = output * (1 - output)
    d_loss_dw = error * d_output_dz * x  # Chain Rule in action
    
    # 4. Update Weights (Gradient Descent)
    w = w - (learning_rate * d_loss_dw)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Output: {output:.4f}")

print(f"\nFinal Weights: {w}")
print(f"Final Output: {output}")

# The Activation Function (Sigmoid)
#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))

#def sigmoid_derivative(x):
#    return sigmoid(x) * (1 - sigmoid(x))

#y = sigmoid(z)
#dy_dz = sigmoid_derivative(z)

#print(f"Input: {x}")
#print(f"Weights: {w}")
#print(f"Bias: {b}")
#print(f"Linear Output (z): {z:.4f}")
#print(f"Activated Output (y): {y:.4f}")
#print(f"Derivative of Sigmoid at z (dy/dz): {dy_dz:.4f}")