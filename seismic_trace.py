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

#Reflectivity series parameters
duration = 1.0  # Duration of the reflectivity series in seconds

reflection_times = [0.15, 0.35, 0.55, 0.75]  # Times of reflection events in seconds
reflection_coefficients = [0.6, -0.4, 0.8, -0.5]  # Reflection coefficients

trace_time, reflectivity_series = create_reflectivity_series(duration, sampling_interval,
                                                              reflection_times, 
                                                              reflection_coefficients)

synthetic_trace = np.convolve(reflectivity_series, wavelet, mode='same') #mode='same' 
#ensures the output (synthetic_trace) has the same length as the input reflectivity series



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
plt.show()
