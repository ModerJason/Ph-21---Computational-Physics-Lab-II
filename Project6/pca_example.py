# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Ensure the Figures directory exists
figures_dir = "Figures"
os.makedirs(figures_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define simulation parameters
fs = 44100  # Sampling frequency in Hz (typical for audio)
T = 10      # Duration in seconds
N = T * fs  # Total number of samples
t = np.linspace(0, T, N)  # Time array for the signal

# Function to generate synthetic audio signals where PCA naturally separates three modes
def generate_audio_signals(num_sensors=3):
    """
    Generate a set of time series (audio signals) where PCA separates:
    1. A pure sine wave (440 Hz).
    2. A chirp signal (100 Hz to 2000 Hz).
    3. Independent noise.

    Parameters:
        num_sensors (int): Number of sensors recording the signals.

    Returns:
        np.ndarray: Array of shape (num_sensors, N) with the generated signals.
    """
    # Fundamental signals
    sine_wave    = 0.25*np.sin(2 * np.pi * 440 * t)  # 440 Hz pure tone
    chirp_signal = np.sin(2 * np.pi * (100 + (1900 / T) * t) * t)  # Chirp from 100 Hz to 2000 Hz

    signals = []  # Store each sensor's time series

    for i in range(num_sensors):
        # Each sensor gets a mix of the sine wave, chirp, and independent noise
        sensor_signal = (
            np.random.uniform(0.8, 1.2) * sine_wave +  # Weighted sine wave
            np.random.uniform(0.8, 1.2) * chirp_signal +  # Weighted chirp
            np.random.normal(0, 0.1, N)  # Independent Gaussian noise
        )
        signals.append(sensor_signal)

    return np.array(signals)  # Return as a NumPy array

# Generate synthetic multi-sensor audio data
num_sensors = 3
audio_data = generate_audio_signals(num_sensors=num_sensors)

# Function to perform PCA manually using NumPy
def pca_numpy(data):
    """
    Perform Principal Component Analysis (PCA) using NumPy.
    """
    mean = np.mean(data, axis=1, keepdims=True)
    centered_data = data - mean

    # Principal components are given by eigenvalues of covariance matrix, or equivalently eigenvalues of X transpose X, where X is design matrix.
    covariance_matrix = np.cov(centered_data)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sorting eigenvalues in descending orders
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    # Project data onto principal components. Each row of projected_data corresponds to a principal component.
    # Each column of projected_data is a transformed data point in this new space.
    projected_data = np.dot(eigenvectors.T, centered_data)
    return projected_data, eigenvalues, eigenvectors

# Apply PCA
pca_transformed, eigenvalues, eigenvectors = pca_numpy(audio_data)

# Compute PSDs for original sensor signals
psd_originals = [welch(audio_data[i], fs=fs, nperseg=fs//2) for i in range(num_sensors)]
freqs_original = psd_originals[0][0]
psd_values_original = np.array([psd[1] for psd in psd_originals])

# Compute PSD for the first three principal components
num_pcs_to_plot = 3
freqs_pca, psd_values_pca = welch(pca_transformed[:num_pcs_to_plot], fs=fs, nperseg=fs//2)

# Use an improved color scheme
#plt.style.use("seaborn-darkgrid")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Create a 2x2 grid with shared axes where meaningful
fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharex="col")

# Top-left: Original sensor time series (first second) - LINEAR SCALE
for i in range(num_sensors):
    ax[0, 0].plot(t[:fs], audio_data[i, :fs], color=colors[i], alpha=0.7, label=f"Sensor {i+1}")
ax[0, 0].set_title("Original Audio Time Series (First Second)", fontsize=14, fontweight='bold')
ax[0, 0].set_ylabel("Amplitude", fontsize=12)
ax[0, 0].legend(fontsize=10, loc="upper right")
ax[0, 0].grid(True, linestyle="--", alpha=0.6)

# Top-right: PSD of original signals (log-log scale)
for i in range(num_sensors):
    ax[0, 1].loglog(freqs_original, psd_values_original[i], color=colors[i], alpha=0.8, label=f"Sensor {i+1}")
ax[0, 1].set_title("Power Spectral Density of Original Signals", fontsize=14, fontweight='bold')
ax[0, 1].set_ylabel("Power Spectral Density", fontsize=12)
ax[0, 1].legend(fontsize=10, loc="upper right")
ax[0, 1].grid(True, which="both", linestyle="--", alpha=0.6)

# Bottom-left: First three principal components (first second) - LINEAR SCALE
for i in range(num_pcs_to_plot):
    ax[1, 0].plot(t[:fs], pca_transformed[i, :fs], color=colors[i], alpha=0.7, label=f"PC {i+1}")
ax[1, 0].set_title("First Three Principal Components (PCA)", fontsize=14, fontweight='bold')
ax[1, 0].set_xlabel("Time [s]", fontsize=12)
ax[1, 0].set_ylabel("Amplitude", fontsize=12)
ax[1, 0].legend(fontsize=10, loc="upper right")
ax[1, 0].grid(True, linestyle="--", alpha=0.6)

# Bottom-right: PSD of principal components (log-log scale)
for i in range(num_pcs_to_plot):
    ax[1, 1].loglog(freqs_pca, psd_values_pca[i], color=colors[i], alpha=0.8, label=f"PC {i+1}")
ax[1, 1].set_title("Power Spectral Density of Principal Components", fontsize=14, fontweight='bold')
ax[1, 1].set_xlabel("Frequency [Hz]", fontsize=12)
ax[1, 1].set_ylabel("Power Spectral Density", fontsize=12)
ax[1, 1].legend(fontsize=10, loc="upper right")
ax[1, 1].grid(True, which="both", linestyle="--", alpha=0.6)

# Improve layout and save as PDF instead of displaying
plt.tight_layout()
pdf_path = os.path.join(figures_dir, "PCA_Audio_Analysis.pdf")
plt.savefig(pdf_path, format="pdf", dpi=300)
plt.close()

# Output the file path so the user knows where the PDF is saved
print(f"Plot saved as: {pdf_path}")
