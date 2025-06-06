import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal as sig
import os

# Create output directory for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load data
print("Loading data for advanced analysis...")
signal_df = pd.read_csv("ecg_fourier_output.csv")
coef_df = pd.read_csv("ecg_coefficients.csv")

# Extract data
original = signal_df["original"].values
noisy = signal_df["noisy"].values
l2_approx = signal_df["l2"].values
l1_approx = signal_df["l1"].values
time = signal_df["n"].values

# Calculate residuals
l2_residual = original - l2_approx
l1_residual = original - l1_approx

# Plot 1: Residual Analysis
plt.figure(figsize=(14, 10))
gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

# Original signal and approximations
ax1 = plt.subplot(gs[0])
ax1.plot(time, original, label="Original", linewidth=2)
ax1.plot(time, l2_approx, label="L2 Approximation", linestyle='--')
ax1.plot(time, l1_approx, label="L1 Approximation", linestyle=':')
ax1.set_title("ECG Signal and Approximations", fontsize=16)
ax1.set_ylabel("Amplitude")
ax1.legend()

# L2 residuals
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(time, l2_residual, color='orange', linewidth=1.5)
ax2.set_title("L2 Approximation Residuals", fontsize=14)
ax2.set_ylabel("Residual")
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# L1 residuals
ax3 = plt.subplot(gs[2], sharex=ax1)
ax3.plot(time, l1_residual, color='green', linewidth=1.5)
ax3.set_title("L1 Approximation Residuals", fontsize=14)
ax3.set_xlabel("Sample Index (n)")
ax3.set_ylabel("Residual")
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig("plots/residual_analysis.png", dpi=300)
plt.show()

# Plot 2: Error distribution histograms
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(l2_residual, bins=50, alpha=0.7, color='orange')
plt.title("L2 Approximation Error Distribution", fontsize=14)
plt.xlabel("Error Magnitude")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(l1_residual, bins=50, alpha=0.7, color='green')
plt.title("L1 Approximation Error Distribution", fontsize=14)
plt.xlabel("Error Magnitude")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("plots/error_distribution.png", dpi=300)
plt.show()

# Plot 3: Time-Frequency Analysis
# Short-time Fourier transform (STFT)
fs = 250  # Assume 250 Hz sampling rate
f, t, Zxx = sig.stft(original, fs, nperseg=128, noverlap=100)

plt.figure(figsize=(12, 8))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude - ECG Signal', fontsize=16)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.savefig("plots/time_frequency_analysis.png", dpi=300)
plt.show()

# Plot 4: Coefficient Phase Analysis
plt.figure(figsize=(14, 6))
# Calculate phases from complex coefficients
l1_phases = np.arctan2(coef_df["l1_imag"].values, coef_df["l1_real"].values) * 180 / np.pi
l2_phases = np.arctan2(coef_df["l2_imag"].values, coef_df["l2_real"].values) * 180 / np.pi

plt.subplot(2, 1, 1)
plt.bar(coef_df["k"] - 0.2, coef_df["l2_magnitude"], width=0.4, label="L2 Magnitude", alpha=0.7)
plt.bar(coef_df["k"] + 0.2, coef_df["l1_magnitude"], width=0.4, label="L1 Magnitude", alpha=0.7)
plt.title("Coefficient Magnitude Comparison", fontsize=14)
plt.xlabel("Coefficient Index (k)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 1, 2)
plt.scatter(coef_df["k"], l2_phases, label="L2 Phase", marker='o', alpha=0.7)
plt.scatter(coef_df["k"], l1_phases, label="L1 Phase", marker='x', alpha=0.7)
plt.title("Coefficient Phase Comparison", fontsize=14)
plt.xlabel("Coefficient Index (k)")
plt.ylabel("Phase (degrees)")
plt.ylim(-180, 180)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Extract phases
l2_phase = np.angle(np.array([complex(r, i) for r, i in zip(coef_df["l2_real"], coef_df["l2_imag"])]))
l1_phase = np.angle(np.array([complex(r, i) for r, i in zip(coef_df["l1_real"], coef_df["l1_imag"])]))

plt.subplot(1, 2, 1)
plt.scatter(coef_df["k"], l2_phase, alpha=0.8, s=50, c='orange', edgecolor='k')
plt.title("L2 Coefficient Phase Analysis", fontsize=14)
plt.xlabel("Coefficient Index (k)")
plt.ylabel("Phase (radians)")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(coef_df["k"], l1_phase, alpha=0.8, s=50, c='green', edgecolor='k')
plt.title("L1 Coefficient Phase Analysis", fontsize=14)
plt.xlabel("Coefficient Index (k)")
plt.ylabel("Phase (radians)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/phase_analysis.png", dpi=300)
plt.show()

print("Advanced analysis plots generated and saved to 'plots' directory!")
