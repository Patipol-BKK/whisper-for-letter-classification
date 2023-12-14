import numpy as np

def signal_power(signal):
	return np.mean(signal ** 2)


def add_noise(signal, noise, snr):
	# Randomly choose where to slice the noise sample
	noise_start = np.random.randint(0, len(noise) - len(signal))
	noise_slice = noise[noise_start:noise_start + len(signal)]

	signal_pwr = signal_power(signal)	
	noise_pwr = signal_power(noise)

	target_noise_pwr =  signal_pwr / (10 ** (snr / 10))

	if noise_pwr > 0:
		# Scale the noise to achieve the desired SNR
		scaled_noise = np.sqrt(target_noise_pwr / noise_pwr) * noise_slice

		# Add the scaled noise to the signal
		noisy_signal = signal + scaled_noise
	else:
		return signal

	return noisy_signal