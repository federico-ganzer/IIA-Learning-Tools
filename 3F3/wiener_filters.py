import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener, welch, hann
from scipy.fft import fft, ifft

def local_wiener_filter(signal, window_size, noise_var = None):
    
    signal = np.array(signal)
    n = len(signal)
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2
    
    if noise_var is None:
        noise_var = np.var(np.diff(signal))/2.0
        
    for i in range(n):
        start = max(0, i- half_window)
        end = min(n, i + half_window + 1)
        window = signal[start:end]
        
        local_mean = np.mean(window)
        local_var = np.var(window)
        
        if local_var > noise_var:
            filtered_signal[i] = local_mean + (local_var - noise_var) / local_var * (signal[i] - local_mean)
            
        else:
            filtered_signal[i] = local_mean
            
    return filtered_signal

def global_wiener_filter(signal, nperseg, noise_var = None):
    signal = np.array(signal)
    n = len(signal)
    if nperseg is None:
        nperseg = max(256, n // 4)
        
    freqs, x_psd = welch(signal, nperseg= nperseg, window= 'hann')
    
    if noise_var is None:
        high_freq_idx = int(0.8 * len(x_psd))
        noise_psd = np.median(x_psd[high_freq_idx:])
    else:
        noise_psd = noise_var

    signal_psd = np.maximum(x_psd - noise_psd, 0)
    h_freq = signal_psd / np.maximum(x_psd, 1e-10)
    
    fft_freqs = np.fft.rfftfreq(n, d=1.0)  # normalized frequencies
    
    
    h_fft = np.interp(fft_freqs, freqs / (freqs[-1] if freqs[-1] > 0 else 1), h_freq)
    
    
    X = np.fft.rfft(signal)
    Y = X * h_fft
    output = np.fft.irfft(Y, n=n)
    
    return output
  
fs = 500
T = 2.0
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Clean signal: 5 Hz + 15 Hz
clean = 1.0 * np.sin(2 * np.pi * 5 * t) + 0.7 * np.sin(2 * np.pi * 15 * t)

# Add white Gaussian noise
np.random.seed(42)
noise_std = 0.4
noise = noise_std * np.random.randn(len(t))
noisy = clean + noise
true_noise_var = noise_std ** 2

# Apply filters
print("Computing filters...")
y_local = local_wiener_filter(noisy, window_size=15, noise_var=true_noise_var)
y_global = global_wiener_filter(noisy, noise_var=true_noise_var, nperseg=256)

# Compute errors
mse_noisy = np.mean((clean - noisy) ** 2)
mse_local = np.mean((clean - y_local) ** 2)
mse_global = np.mean((clean - y_global) ** 2)

print(f"MSE (noisy)        : {mse_noisy:.6f}")
print(f"MSE (local Wiener) : {mse_local:.6f}")
print(f"MSE (global Wiener): {mse_global:.6f}")
print(f"SNR improvement (local) : {10*np.log10(mse_noisy/mse_local):.2f} dB")
print(f"SNR improvement (global): {10*np.log10(mse_noisy/mse_global):.2f} dB")

# Plot: full signal
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(t, clean, 'k-', linewidth=2, label='Clean signal', zorder=3)
ax.plot(t, noisy, 'b-', alpha=0.3, linewidth=0.8, label='Noisy signal', zorder=1)
ax.set_ylabel('Amplitude')
ax.set_title('Input Signal')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t, clean, 'k-', linewidth=2, label='Clean signal')
ax.plot(t, y_local, 'r-', linewidth=1.5, label='Local Wiener filter', alpha=0.8)
ax.set_ylabel('Amplitude')
ax.set_title(f'Local Wiener (window=15, MSE={mse_local:.6f})')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t, clean, 'k-', linewidth=2, label='Clean signal')
ax.plot(t, y_global, 'g-', linewidth=1.5, label='Global Wiener–Hopf (FFT)', alpha=0.8)
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time [s]')
ax.set_title(f'Global Wiener–Hopf (MSE={mse_global:.6f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wiener_comparison_time.png', dpi=100)
plt.show()

# Plot: zoomed-in view (first 0.5 seconds)
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
t_zoom = t < 0.5

ax = axes[0]
ax.plot(t[t_zoom], clean[t_zoom], 'k-', linewidth=2, label='Clean signal', zorder=3)
ax.plot(t[t_zoom], noisy[t_zoom], 'b-', alpha=0.3, linewidth=0.8, label='Noisy signal', zorder=1)
ax.set_ylabel('Amplitude')
ax.set_title('Input Signal (zoomed)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t[t_zoom], clean[t_zoom], 'k-', linewidth=2, label='Clean signal')
ax.plot(t[t_zoom], y_local[t_zoom], 'r-', linewidth=1.5, label='Local Wiener', alpha=0.8)
ax.plot(t[t_zoom], noisy[t_zoom], 'b-', alpha=0.2, linewidth=0.5)
ax.set_ylabel('Amplitude')
ax.set_title('Local Wiener (zoomed)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t[t_zoom], clean[t_zoom], 'k-', linewidth=2, label='Clean signal')
ax.plot(t[t_zoom], y_global[t_zoom], 'g-', linewidth=1.5, label='Global Wiener–Hopf', alpha=0.8)
ax.plot(t[t_zoom], noisy[t_zoom], 'b-', alpha=0.2, linewidth=0.5)
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time [s]')
ax.set_title('Global Wiener–Hopf (zoomed)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wiener_comparison_zoom.png', dpi=100)
plt.show()

# Plot: frequency domain
freqs, pxx_noisy = welch(noisy, fs=fs, nperseg=512)
freqs, pxx_clean = welch(clean, fs=fs, nperseg=512)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(freqs, pxx_clean, 'k-', linewidth=2, label='Clean signal PSD')
ax.semilogy(freqs, pxx_noisy, 'b-', linewidth=1, alpha=0.7, label='Noisy signal PSD')
ax.axhline(true_noise_var, color='r', linestyle='--', linewidth=1, label=f'Noise variance = {true_noise_var:.4f}')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Power Spectral Density')
ax.set_title('Power Spectral Density (Welch)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('wiener_psd.png', dpi=100)
plt.show()