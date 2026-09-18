import numpy as np
import matplotlib.pyplot as plt

def ricker_wavelet(f, length, dt):
    "Generate a zero phase Ricker wavelet"

    n_samples = int(length / dt)

    time = np.linspace(-length / 2, length / 2, n_samples)  

    pi_f_time = np.pi * f * time

    amplitude = (1 - 2 * pi_f_time ** 2) * np.exp(-pi_f_time ** 2)

    return time, amplitude

#Wavelet parameters
dom_freq = 25  # Dominant frequency in Hz
wavelet_length = 0.2  # Length of the wavelet in seconds
sampling_interval = 0.001  # Sampling interval in seconds

time, wavelet = ricker_wavelet(dom_freq, wavelet_length, sampling_interval) 

#Plot the wavelet
plt.figure(figsize=(10, 5))
plt.plot(time * 1000, wavelet, label='Ricker Wavelet', color='blue')

plt.title('Ricker Wavelet')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()