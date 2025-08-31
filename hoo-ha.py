import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. Data Generation: Two distinct sine waves ---
def generate_data(num_samples=2000, seq_len=100):
    """Generates two classes of sine waves."""
    X = np.zeros((num_samples, seq_len))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        freq = np.random.uniform(1.0, 3.0) if i % 2 == 0 else np.random.uniform(5.0, 8.0)
        phase = np.random.uniform(0, np.pi)
        t = np.linspace(0, 1, seq_len)
        X[i, :] = np.sin(2 * np.pi * freq * t + phase) + np.random.normal(0, 0.1, seq_len)
        y[i] = 0 if i % 2 == 0 else 1
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- 2. Model Definition: A simple MLP ---
# The paper recommends a simple MLP with Tanh activations for clearer patterns [cite: 185, 186]
class SimpleMLP(nn.Module):
    def __init__(self, input_size=100, hidden_size1=512, hidden_size2=256, output_size=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 3. Holographic Analysis ---
# Class structure and metrics are taken directly from the research paper's framework [cite: 171, 174]
class HolographicAnalyzer:
    def fourier_analysis(self, weight_matrix):
        """Performs Fourier analysis on a weight matrix."""
        if isinstance(weight_matrix, torch.Tensor):
            weight_matrix = weight_matrix.detach().cpu().numpy()
            
        # 2D Fast Fourier Transform
        fft_result = fft2(weight_matrix)
        
        # Calculate Power Spectrum
        power_spectrum = np.abs(fft_result)**2
        
        # Coherence Ratio: Measures how 'peaky' the spectrum is. 
        # A high ratio suggests information is stored in specific frequencies (interference pattern).
        coherence_ratio = np.max(power_spectrum) / np.sum(power_spectrum)
        
        return {
            'power_spectrum': fftshift(power_spectrum), # Shift zero-frequency to the center
            'coherence_ratio': coherence_ratio,
        }
        
    def analyze_and_plot(self, model, layer_name='fc1.weight', title=''):
        """Helper function to run analysis and plot results."""
        weights = dict(model.named_parameters())[layer_name]
        analysis = self.fourier_analysis(weights)
        
        print(f"{title} - Coherence Ratio: {analysis['coherence_ratio']:.6f}")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log(analysis['power_spectrum'] + 1e-9), cmap='viridis') # log scale for visibility
        plt.title(title)
        plt.colorbar(label='Log Power')
        plt.show()
        
        return analysis

# --- 4. Damage Function ---
def apply_damage(model, damage_ratio=0.5):
    """Zeros out a random fraction of weights to test for graceful degradation."""
    print(f"\nApplying {damage_ratio*100}% damage to all weights...")
    with torch.no_grad():
        for param in model.parameters():
            mask = (torch.rand_like(param) > damage_ratio).float()
            param.data *= mask
    print("Damage applied.")
    return model

# --- 5. Main Execution ---
if __name__ == '__main__':
    # Setup
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SimpleMLP()
    analyzer = HolographicAnalyzer()
    
    # Analyze weights of an UNTRAINED model
    print("--- Analysis of Untrained Model ---")
    analyzer.analyze_and_plot(model, title='Power Spectrum of Untrained Weights')

    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n--- Training Model ---")
    for epoch in range(100): # Short training for demonstration
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

    # Analyze weights of the TRAINED model
    print("\n--- Analysis of Trained Model (Before Damage) ---")
    analyzer.analyze_and_plot(model, title='Power Spectrum of Trained Weights')

    # Test performance before damage
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"\nAccuracy before damage: {accuracy * 100:.2f}%")

    # Apply damage and re-analyze
    model = apply_damage(model, damage_ratio=0.75) # Heavy 75% damage
    print("\n--- Analysis of Trained Model (After 75% Damage) ---")
    analyzer.analyze_and_plot(model, title='Power Spectrum of Damaged Weights')

    # Test performance after damage
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"\nAccuracy after 75% damage: {accuracy * 100:.2f}%")