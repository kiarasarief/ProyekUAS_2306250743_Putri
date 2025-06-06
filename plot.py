import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load data
print("Loading data...")
signal_df = pd.read_csv("ecg_fourier_output.csv")
spectrum_df = pd.read_csv("ecg_spectrum.csv")
coef_df = pd.read_csv("ecg_coefficients.csv")

# Plot 1: Original vs Approximations
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top plot: Full signal
axes[0].plot(signal_df["n"], signal_df["original"], label="Original", linewidth=2)
axes[0].plot(signal_df["n"], signal_df["l2"], label="L2 Approximation", linestyle='--')
axes[0].plot(signal_df["n"], signal_df["l1"], label="L1 Approximation", linestyle=':')
axes[0].set_title("ECG Signal Decomposition using Fourier Transform", fontsize=16)
axes[0].set_ylabel("Amplitude")
axes[0].legend()

# Bottom plot: Zoomed in on QRS complex
zoom_start = 250
zoom_end = 350
axes[1].plot(signal_df["n"][zoom_start:zoom_end], signal_df["original"][zoom_start:zoom_end], label="Original", linewidth=2)
axes[1].plot(signal_df["n"][zoom_start:zoom_end], signal_df["l2"][zoom_start:zoom_end], label="L2 Approximation", linestyle='--')
axes[1].plot(signal_df["n"][zoom_start:zoom_end], signal_df["l1"][zoom_start:zoom_end], label="L1 Approximation", linestyle=':')
axes[1].set_title("Zoomed View of QRS Complex", fontsize=16)
axes[1].set_xlabel("Sample Index (n)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/ecg_decomposition.png", dpi=300)
plt.show()

# Plot 2: Analysis with Noisy Signal
plt.figure(figsize=(14, 8))
plt.plot(signal_df["n"], signal_df["original"], label="Original", linewidth=2)
plt.plot(signal_df["n"], signal_df["noisy"], label="Noisy", alpha=0.5, color='gray')
plt.plot(signal_df["n"], signal_df["l2"], label="L2 Approximation", linestyle='--')
plt.plot(signal_df["n"], signal_df["l1"], label="L1 Approximation", linestyle=':')
plt.title("ECG Signal Decomposition with Noise", fontsize=16)
plt.xlabel("Sample Index (n)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("plots/ecg_with_noise.png", dpi=300)
plt.show()

# Plot 3: Power Spectrum
plt.figure(figsize=(14, 6))
# Only plot the first half of the spectrum (up to Nyquist frequency)
N = len(spectrum_df)
nyquist = N // 2
plt.semilogy(spectrum_df["k"][:nyquist], spectrum_df["power"][:nyquist])
plt.title("Power Spectrum of ECG Signal", fontsize=16)
plt.xlabel("Frequency Bin (k)")
plt.ylabel("Power (log scale)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/ecg_spectrum.png", dpi=300)
plt.show()

# Plot 4: Coefficient Magnitudes Comparison
plt.figure(figsize=(14, 6))
plt.bar(coef_df["k"] - 0.2, coef_df["l2_magnitude"], width=0.4, label="L2 Coefficients", alpha=0.7)
plt.bar(coef_df["k"] + 0.2, coef_df["l1_magnitude"], width=0.4, label="L1 Coefficients", alpha=0.7)
plt.title("Fourier Coefficient Magnitudes Comparison", fontsize=16)
plt.xlabel("Coefficient Index (k)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/coefficient_comparison.png", dpi=300)
plt.show()

print("All plots generated and saved to 'plots' directory!")
