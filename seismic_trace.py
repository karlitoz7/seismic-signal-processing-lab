from operator import index

import numpy as np
import matplotlib.pyplot as plt

def ricker_wavelet(f, length, dt):
    """Generate a zero phase Ricker wavelet"""

    n_samples = int(length / dt) + 1

    time = np.linspace(-length / 2, length / 2, n_samples)  

    pi_f_time = np.pi * f * time

    amplitude = (1 - 2 * pi_f_time ** 2) * np.exp(-pi_f_time ** 2)

    return time, amplitude

#Wavelet parameters
dom_freq = 25  # Dominant frequency in Hz
wavelet_length = 0.2  # Length of the wavelet in seconds
sampling_interval = 0.001  # Sampling interval in seconds

time, wavelet = ricker_wavelet(dom_freq, wavelet_length, sampling_interval) 

print(f"Number of samples: {len(time)}")
print(f"Sample interval: {(time[1] - time[0]) * 1000:.3f} ms")
print(f"Peak Amplitude: {wavelet.max():.3f}")

def create_reflectivity_series(duration, dt, reflection_times, reflection_coefficients):

    """Create a synthetic reflectivity series with random spikes"""

    n_samples = int(round(duration / dt)) + 1
    time = np.linspace(0, duration, n_samples)
    reflectivity = np.zeros(n_samples)

    if len(reflection_times) != len(reflection_coefficients):
        raise ValueError("Reflection times and coefficients must have equal lengths.")

    for event_time, coefficient in zip(reflection_times, reflection_coefficients):
        sample_index = int(round(event_time / dt))
        reflectivity[sample_index] = coefficient

    return time, reflectivity

def add_white_noise(signal, snr_db, seed=None):
    """Add white Gaussian noise to a signal based on a specified SNR in dB"""

    random_generator = np.random.default_rng(seed) # Use a random generator for reproducibility

    signal_power = np.mean(signal ** 2)  

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_standard_deviation = np.sqrt(noise_power)

    noise = random_generator.normal(loc=0.0, scale=noise_standard_deviation, size=signal.shape)

    noisy_signal = signal + noise

    return noisy_signal, noise

def amplitude_spectrum(signal, dt):
    """Calculate the one-sided amplitude spectrum of a signal"""

    n_samples = len(signal)

    frequencies = np.fft.rfftfreq(n_samples, dt)

    complex_spectrum = np.fft.rfft(signal)
    amplitude = np.abs(complex_spectrum) / n_samples

    #Convert to a one-sided amplitude spectrum
    if n_samples % 2 == 0:  # Even number of samples
        amplitude[1:-1] *= 2
    else:  # Odd number of samples
        amplitude[1:] *= 2

        return frequencies, amplitude
    
#Reflectivity series parameters
duration = 1.0  # Duration of the reflectivity series in seconds

reflection_times = [0.15, 0.35, 0.55, 0.75]  # Times of reflection events in seconds
reflection_coefficients = [0.6, -0.4, 0.8, -0.5]  # Reflection coefficients

trace_time, reflectivity_series = create_reflectivity_series(duration, sampling_interval,
                                                              reflection_times, 
                                                              reflection_coefficients)

synthetic_trace = np.convolve(reflectivity_series, wavelet, mode='same') #mode='same' 
#ensures the output (synthetic_trace) has the same length as the input reflectivity series

# Add noise to the synthetic trace
target_snr_db = 10.0 #SNR in dB

noisy_trace, noise = add_white_noise(synthetic_trace, target_snr_db, seed=42)

#Calculate the amplitude spectra of the three signals
wavelet_frequencies, wavelet_amplitude = amplitude_spectrum(wavelet, sampling_interval)
clean_frequencies, clean_amplitude = amplitude_spectrum(synthetic_trace, sampling_interval)
noisy_frequencies, noisy_amplitude = amplitude_spectrum(noisy_trace, sampling_interval)

#Estimate the wavelet's dominant freuency 
peak_index = np.argmax(wavelet_amplitude[1:]) + 1  # Exclude the DC component at index 0
measured_dominant_frequency = wavelet_frequencies[peak_index]

print(f"Measured dominant frequency of the wavelet: {measured_dominant_frequency:.2f} Hz")

actual_snr_db = 10 * np.log10(np.mean(synthetic_trace ** 2) / np.mean(noise ** 2)) #Calculate the SNR actually obtained
print(f"Target SNR: {target_snr_db:.2f} dB") 
print(f"Actual SNR: {actual_snr_db:.2f} dB")

#Plot the wavelet
plt.figure(figsize=(10, 5))
plt.plot(time * 1000, wavelet, label='Ricker Wavelet', color='blue')

plt.title('Ricker Wavelet')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()

#Plot the reflectivity series
plt.figure(figsize=(10, 5))
plt.stem(trace_time * 1000, reflectivity_series, basefmt='gray', label='Reflectivity Series')
plt.title("Synthetic Reflectivity Series")
plt.xlabel("Two-way time (ms)")
plt.ylabel("Reflection coefficient")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Compare reflectivity series and synthetic trace
figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].axhline(y=0, color='gray', linewidth=1)
axes[0].stem(np.array(reflection_times) * 1000, reflection_coefficients, basefmt=' ')
axes[0].set_xlim(0, duration * 1000)
axes[0].set_title("Synthetic Reflectivity Series")
axes[0].set_ylabel("Reflection coefficient")
axes[0].grid(True, alpha=0.3)

axes[1].plot(trace_time * 1000, synthetic_trace, label='Synthetic Trace', color='orange')
axes[1].set_title("Synthetic Seismic Trace")
axes[1].set_xlabel("Two-way time (ms)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)

figure.tight_layout()
figure.savefig("figures/synthetic_trace.png", dpi=300, bbox_inches='tight')

noise_figure, noise_axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
noise_axes[0].plot(trace_time * 1000, synthetic_trace, color='darkred')
noise_axes[1].plot(trace_time * 1000, noisy_trace, label='Noise', color='steelblue')
noise_axes[0].set_title("Clean Synthetic Trace")
noise_axes[0].set_ylabel("Amplitude")
noise_axes[0].grid(True, alpha=0.3)
noise_axes[1].set_title(f"Noisy Synthetic Trace - SNR = {target_snr_db:.0f} dB")
noise_axes[1].set_xlabel("Two-way time (ms)")
noise_axes[1].set_ylabel("Amplitude")
noise_axes[1].grid(True, alpha=0.3)

noise_figure.tight_layout()

noise_figure.savefig("figures/noise_comparison.png", dpi=300, bbox_inches='tight')

spectrum_figure, spectrum_axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#Wavelet spectrum
spectrum_axes[0].plot(wavelet_frequencies, wavelet_amplitude, color='purple', linewidth=2, label='Ricker Wavelet')
spectrum_axes[0].axvline(dom_freq, color='black', linestyle='--', alpha=0.7, label=f"{dom_freq} Hz")
spectrum_axes[0].set_title("Amplitude Spectrum of Ricker Wavelet")
spectrum_axes[0].set_ylabel("Amplitude")
spectrum_axes[0].legend()
spectrum_axes[0].grid(True, alpha=0.3)


#Clean and noisy trace spectra
spectrum_axes[1].plot(clean_frequencies, clean_amplitude, color='orange', linewidth=2, label='Clean Synthetic Trace')
spectrum_axes[1].plot(noisy_frequencies, noisy_amplitude, color='steelblue', linewidth=2, label='Noisy Synthetic Trace')
spectrum_axes[1].set_title("Amplitude Spectra of Synthetic Traces")
spectrum_axes[1].set_xlabel("Frequency (Hz)")
spectrum_axes[1].set_ylabel("Amplitude")
spectrum_axes[1].set_xlim(0, 100)
spectrum_axes[1].legend()
spectrum_axes[1].grid(True, alpha=0.3)

spectrum_figure.tight_layout()

spectrum_figure.savefig("figures/amplitude_spectra.png", dpi=300, bbox_inches='tight')

plt.show()