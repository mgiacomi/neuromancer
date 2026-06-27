import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Custom feedforward neural network class using NumPy
class BreastCancerNN:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.2, random_seed=42):
        # Set seed for reproducibility
        np.random.seed(random_seed)
        
        # Initialize weights and biases
        # He initialization for ReLU activation in the hidden layer
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        # Xavier initialization for Sigmoid activation in the output layer
        self.w2 = np.random.randn(hidden_dim, 1) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))
        
        self.lr = learning_rate
        
    def relu(self, z):
        return np.maximum(0, z)
        
    def relu_derivative(self, z):
        return (z > 0).astype(float)
        
    def sigmoid(self, z):
        # Clip values to avoid overflow/underflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
        
    def forward(self, X):
        # Input to Hidden Layer
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Hidden to Output Layer
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
        
    def backward(self, X, y):
        m = X.shape[0]
        
        # Output layer gradient (Binary Cross-Entropy w.r.t Sigmoid logit is simply a2 - y)
        d_z2 = self.a2 - y
        d_w2 = np.dot(self.a1.T, d_z2) / m
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / m
        
        # Hidden layer gradient
        d_z1 = np.dot(d_z2, self.w2.T) * self.relu_derivative(self.z1)
        d_w1 = np.dot(X.T, d_z1) / m
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / m
        
        # Parameter updates (Gradient Descent)
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Clip predictions to prevent absolute log(0) which results in NaN
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss

def main():
    print("=" * 60)
    print("      Breast Cancer Classifier Neural Network (from scratch)      ")
    print("=" * 60)
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'breast_cancer_data.csv')
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    
    print(f"Dataset Dimensions: {df.shape[0]} samples, {df.shape[1] - 1} features")
    
    # 2. Split Data: 75% training, 25% testing
    # Using stratify=y keeps target class proportions consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y
    )
    print(f"Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    # 3. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features successfully scaled using StandardScaler.")
    
    # 4. Initialize Neural Network
    input_dim = X_train_scaled.shape[1]
    hidden_dim = 16
    learning_rate = 0.1
    epochs = 10000
    
    nn = BreastCancerNN(input_dim=input_dim, hidden_dim=hidden_dim, learning_rate=learning_rate)
    print(f"\nCreated Neural Network: Input Dimension = {input_dim}, Hidden Neurons = {hidden_dim}, LR = {learning_rate}")
    
    # 5. Training Loop
    print("\nTraining network...")
    print("-" * 75)
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Train Acc':<15}{'Test Loss':<15}{'Test Acc':<15}")
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        # Forward Pass on Train
        train_pred = nn.forward(X_train_scaled)
        train_loss = nn.compute_loss(y_train, train_pred)
        
        # Backpropagation
        nn.backward(X_train_scaled, y_train)
        
        # Evaluate on Test (without backprop)
        test_pred = nn.forward(X_test_scaled)
        test_loss = nn.compute_loss(y_test, test_pred)
        
        # Calculate Accuracies
        train_acc = np.mean((train_pred >= 0.5) == y_train)
        test_acc = np.mean((test_pred >= 0.5) == y_test)
        
        # Print progress every 100 epochs
        if epoch % 100 == 0 or epoch == 1:
            print(f"{epoch:<8}{train_loss:<15.4f}{train_acc*100:<13.2f}%{test_loss:<15.4f}{test_acc*100:<13.2f}%")
            
    # 6. Final Evaluation
    print("-" * 75)
    print("\nTraining complete! Evaluating final model on Test Set...")
    
    final_test_pred_prob = nn.forward(X_test_scaled)
    final_test_pred = (final_test_pred_prob >= 0.5).astype(int)
    
    test_accuracy = np.mean(final_test_pred == y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, final_test_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, final_test_pred, target_names=["Cancer (0)", "No Cancer (1)"]))

if __name__ == "__main__":
    main()
