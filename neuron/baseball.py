import math
import random

# 1. Activation Function (Sigmoid squashes numbers between 0 and 1)
def sigmoid(x):
    # Bound input to avoid overflow in math.exp
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 2. Probability function to generate realistic training data based on rules
def calculate_play_probability(temp, wind, rain, sun, lightning):
    # Temperature factor: games mostly played between 40 and 90 degrees.
    # Decays linearly to 0 at -10 and 120.
    if temp < 40:
        temp_factor = (temp - (-10)) / 50.0
    else:
        temp_factor = (120 - temp) / 30.0
    temp_factor = max(0.0, min(1.0, temp_factor))

    # Wind factor: wind <= 15 mph is perfect. Above 15 mph, playing probability
    # decays gradually (exponential decay with k=30).
    if wind <= 15:
        wind_factor = 1.0
    else:
        wind_factor = math.exp(-(wind - 15) / 30.0)

    # Rain factor: rain makes playing very unlikely (15% reduction)
    rain_factor = 0.85 if rain else 1.0

    # Sun factor: sunny days give a slight boost, cloudy days a slight penalty
    sun_factor = 1.1 if sun else 0.9

    # Lightning factor: lightning mostly cancels games (98% reduction)
    lightning_factor = 0.02 if lightning else 1.0

    # Combine factors and clip between 0 and 1
    p = temp_factor * wind_factor * rain_factor * sun_factor * lightning_factor
    return max(0.0, min(1.0, p))


# Scale inputs to [0, 1] range for neural network training
def scale_inputs(temp, wind, rain, sun, lightning):
    # temp: -10 to 120
    temp_scaled = (temp - (-10)) / (120 - (-10))
    # wind: 0 to 100
    wind_scaled = wind / 100.0
    # rain, sun, lightning: already 0 or 1
    rain_val = 1.0 if rain else 0.0
    sun_val = 1.0 if sun else 0.0
    lightning_val = 1.0 if lightning else 0.0
    return [temp_scaled, wind_scaled, rain_val, sun_val, lightning_val]


# Generate 100 rows of test/training data
random.seed(42)  # For reproducible results

training_data = []
print("Generating 100 rows of training data...")
print(f"{'Row':<5}{'Temp (F)':<10}{'Wind (mph)':<12}{'Rain':<8}{'Sun':<8}{'Lightning':<12}{'Prob Play':<12}{'Play Ball?':<12}")
print("-" * 80)

for idx in range(100):
    # 1. Random weather conditions
    temp = random.uniform(-10, 120)
    wind = random.uniform(0, 100)
    
    # Dependencies: rain, sun, lightning are correlated
    rain = random.random() < 0.20  # 20% rain probability
    
    if rain:
        sun = random.random() < 0.10  # 10% chance of sun if raining
        lightning = random.random() < 0.30  # 30% chance of lightning if raining
    else:
        sun = random.random() < 0.70  # 70% chance of sun if dry
        lightning = random.random() < 0.01  # 1% chance of lightning if dry

    # 2. Calculate the target decision with Bernoulli sampling (incorporating noise)
    p_play = calculate_play_probability(temp, wind, rain, sun, lightning)
    target = 1 if random.random() < p_play else 0

    # 3. Store raw and scaled representations
    inputs_scaled = scale_inputs(temp, wind, rain, sun, lightning)
    training_data.append({
        "raw_inputs": (temp, wind, rain, sun, lightning),
        "inputs": inputs_scaled,
        "target": target
    })

    # Print first 10 rows as a representative sample
    if idx < 10:
        print(f"{idx+1:<5}{temp:>8.1f} {wind:>11.1f}   {str(rain):<8}{str(sun):<8}{str(lightning):<12}{p_play:>9.2%}{'YES' if target == 1 else 'NO':>11}")

print("...\n")

# 3. Initialize Weights and Biases randomly
num_inputs = 5
num_hidden = 4

# Hidden Layer (num_hidden neurons, each taking num_inputs inputs)
hidden_weights = [[random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_hidden)]
hidden_biases = [random.uniform(-1, 1) for _ in range(num_hidden)]

# Output Layer (1 Neuron, taking num_hidden inputs from the hidden layer)
output_weights = [random.uniform(-1, 1) for _ in range(num_hidden)]
output_bias = random.uniform(-1, 1)

learning_rate = 0.3
epochs = 5000

print(f"Training the 5-input, {num_hidden}-hidden-neuron umpire network for {epochs} epochs...")

# 4. The Training Loop (Epochs)
for epoch in range(epochs + 1):
    total_loss = 0

    for data in training_data:
        inputs = data["inputs"]
        target = data["target"]

        # --- FORWARD PASS ---
        # Hidden Layer
        hidden_outputs = []
        for i in range(num_hidden):
            z = sum(inputs[j] * hidden_weights[i][j] for j in range(num_inputs)) + hidden_biases[i]
            hidden_outputs.append(sigmoid(z))

        # Output Layer
        z_out = sum(hidden_outputs[i] * output_weights[i] for i in range(num_hidden)) + output_bias
        final_output = sigmoid(z_out)

        # Calculate Error (Loss)
        error = target - final_output
        total_loss += error**2

        # --- BACKPROPAGATION (The Learning Part) ---
        # Output Layer Gradient
        d_output = error * sigmoid_derivative(final_output)

        # Hidden Layer Gradients
        d_hidden = []
        for i in range(num_hidden):
            d_h = d_output * output_weights[i] * sigmoid_derivative(hidden_outputs[i])
            d_hidden.append(d_h)

        # Update Weights and Biases for Output Neuron
        for i in range(num_hidden):
            output_weights[i] += d_output * hidden_outputs[i] * learning_rate
        output_bias += d_output * learning_rate

        # Update Weights and Biases for Hidden Neurons
        for i in range(num_hidden):
            for j in range(num_inputs):
                hidden_weights[i][j] += d_hidden[i] * inputs[j] * learning_rate
            hidden_biases[i] += d_hidden[i] * learning_rate

    # Print progress every 1000 steps
    if epoch % 1000 == 0:
        mean_squared_error = total_loss / len(training_data)
        print(f"Epoch {epoch:4d} - Mean Squared Error: {mean_squared_error:.4f}")

print("\nTraining Complete! Testing the Umpire on Custom Scenarios:")
print("-" * 80)

# 5. Test the Trained Network on New Data
test_days = [
    {"desc": "Beautiful warm dry day", "raw": (75, 5, False, True, False)},
    {"desc": "Freezing cold dry day ", "raw": (-5, 10, False, True, False)},
    {"desc": "Extremely hot dry day  ", "raw": (115, 12, False, True, False)},
    {"desc": "Windy warm dry day     ", "raw": (72, 35, False, True, False)},
    {"desc": "Rainy warm day         ", "raw": (68, 10, True, False, False)},
    {"desc": "Lightning storm day    ", "raw": (72, 8, True, False, True)},
    {"desc": "Cool overcast dry day  ", "raw": (55, 8, False, False, False)},
]

for day in test_days:
    temp, wind, rain, sun, lightning = day["raw"]
    inputs = scale_inputs(temp, wind, rain, sun, lightning)

    # Forward pass
    hidden_outputs = []
    for i in range(num_hidden):
        z = sum(inputs[j] * hidden_weights[i][j] for j in range(num_inputs)) + hidden_biases[i]
        hidden_outputs.append(sigmoid(z))

    z_out = sum(hidden_outputs[i] * output_weights[i] for i in range(num_hidden)) + output_bias
    prediction = sigmoid(z_out)

    decision = "PLAY BALL!" if prediction > 0.5 else "STAY HOME"
    print(
        f"{day['desc']} -> Temp: {temp:>3}°F, Wind: {wind:>2}mph, Rain: {str(rain):<5}, Sun: {str(sun):<5}, Lightning: {str(lightning):<5} "
        f"-> Confidence: {prediction*100:5.1f}% -> Decision: {decision}"
    )