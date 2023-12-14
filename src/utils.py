import numpy as np

def signal_power(signal):
	return np.mean(signal ** 2)


def add_noise(signal, noise, snr, length):
	# Randomly choose where to slice the noise sample
	noise_start = np.random.randint(0, len(noise) - len(signal))
	noise_slice = noise[noise_start:noise_start + length]

	signal_pwr = signal_power(signal)	
	noise_pwr = signal_power(noise)

	target_noise_pwr =  signal_pwr / (10 ** (snr / 10))

	if noise_pwr > 0:
		# Scale the noise to achieve the desired SNR
		scaled_noise = np.sqrt(target_noise_pwr / noise_pwr) * noise_slice

		if len(scaled_noise) > signal:
			noisy_signal = scaled_noise
			noisy_signal[:len(signal)] += signal
		else:
			noisy_signal = signal
			noisy_signal[:len(scaled_noise)] += scaled_noise
	else:
		return signal

	return noisy_signal