import bisect
import random
import os
import numpy as np
import soundfile as sf
import librosa
from transformers import WhisperFeatureExtractor
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from utils import add_noise

CUT_OFF = 150

class AudioDataset(Dataset):

	def __init__(self, folder_dir=None, noise_dir=None, snr_range=(0, 0.5), selected_labels=None, num_samples=None):
		super().__init__()
		self.feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')

		if folder_dir is None or selected_labels is None or num_samples is None:
			return
		labels = os.listdir(folder_dir)
		
		self.label2class = dict()
		self.class2label = dict()
		
		self.samples = []
		self.class_count = np.zeros(len(selected_labels)+1)

		self.snr_range = snr_range
		print('selected_labels')
		if not noise_dir is None:
			self.has_noise = True

			# Read and store noise audio as pairs of lengths and file paths
			self.noise_list = []
			for noise in os.listdir(noise_dir):
				noise_path = os.path.join(noise_dir, noise)
				data, sr = sf.read(noise_path)
				data = data.flatten()
				length = len(data) / sr

				self.noise_list.append((length, noise_path))
			self.noise_list.sort()

		else:
			self.noise_list = []
			self.has_noise = False

		# Process audio files for selected classes
		for idx, selected_label in enumerate(tqdm(selected_labels, desc='Processing Selected Classes')):
			# Add label name to dict for conversion between class id and class names later
			self.label2class[selected_label] = idx
			self.class2label[idx] = selected_label
			
			label_folder_path = os.path.join(folder_dir, selected_label)

			# Check if number of samples selected exceed the number of dataset files
			if num_samples > len(list(os.listdir(label_folder_path))):
				num_samples_local = len(list(os.listdir(label_folder_path)))
			else:
				num_samples_local = num_samples
			
			selected_audio_files = random.sample(list(os.listdir(label_folder_path)), num_samples_local)

			# Process audio for each classes
			for selected_audio_file in selected_audio_files:
				file_path = os.path.join(label_folder_path, selected_audio_file)
				data, sr = sf.read(file_path)
				
				# Convert to 16k sample rate and flatten
				if sr != 16000:
					data = librosa.resample(data, orig_sr=sr, target_sr=16000)
					sr = 16000
				data = data.flatten()

				# Add noise
				if self.has_noise:
					target_snr = random.uniform(self.snr_range[0], self.snr_range[1])
					noisy_signal = add_noise(data, self._get_noise(len(data)/sr), target_snr)

					data = noisy_signal

				# Extract audio features
				features = self.feature_extractor(data, sampling_rate=sr)
				feature_length = self._feature_end(torch.tensor(features['input_features'][0]))
				
				segmented_features = features['input_features'][0][:, :CUT_OFF]
				self.samples.append({
					'features': torch.tensor(segmented_features).cuda(),
					'class_id': torch.tensor(idx).cuda(),
				})
				self.class_count[idx] += 1
		
		# Process audio files for the remaining labels (this is put into 'others' class)
		remaining_labels = [x for x in labels if x not in selected_labels]
		for remaining_label in tqdm(remaining_labels, desc='Processing Misc. Class'):
			self.label2class['other'] = len(selected_label)
			self.class2label[len(selected_label)] = 'other'
			
			label_folder_path = os.path.join(folder_dir, remaining_label)

			if num_samples > len(list(os.listdir(label_folder_path))):
				num_samples_local = len(list(os.listdir(label_folder_path)))
			else:
				num_samples_local = num_samples
			
			selected_audio_files = random.sample(list(os.listdir(label_folder_path)), num_samples_local)

			for selected_audio_file in selected_audio_files:
				file_path = os.path.join(label_folder_path, selected_audio_file)
				data, sr = sf.read(file_path)
				
				if sr != 16000:
					data = librosa.resample(data, orig_sr=sr, target_sr=16000)
					sr = 16000
				data = data.flatten()

				# Add noise
				if self.has_noise:
					target_snr = random.uniform(self.snr_range[0], self.snr_range[1])
					noisy_signal = add_noise(data, self._get_noise(len(data)/sr), target_snr)

					data = noisy_signal

				features = self.feature_extractor(data, sampling_rate=sr)
				feature_length = self._feature_end(torch.tensor(features['input_features'][0]))
				
				segmented_features = features['input_features'][0][:, :CUT_OFF]
				self.samples.append({
					'features': torch.tensor(segmented_features).cuda(),
					'class_id': torch.tensor(len(selected_labels)).cuda(),
				})
				self.class_count[len(selected_labels)] += 1
	def _feature_end(self, feature_tensor):
	    equal_to_first = torch.all(torch.eq(feature_tensor, feature_tensor[0]), dim=0)
	    
	    first_index = torch.nonzero(equal_to_first)[0]
	    
	    return first_index.item() if first_index.numel() > 0 else -1

	def _get_noise(self, min_length):
		index = bisect.bisect_right(self.noise_list, (min_length,))  # Find the index to the right of the threshold
		noise_slice = self.noise_list[index:] if index < len(self.noise_list) else []
		noise_path = random.choice(noise_slice)[1]

		noise_signal, sr = sf.read(noise_path)
		if sr != 16000:
			noise_signal = librosa.resample(noise_signal, orig_sr=sr, target_sr=16000)
			sr = 16000
		noise_signal = noise_signal.flatten()
		return noise_signal
		
	def __getitem__(self, idx):
		return self.samples[idx]['features'], self.samples[idx]['class_id']
	
	def __len__(self):
		return len(self.samples)

	def save(self, filename):
		np.savez(filename, self.samples, self.label2class, self.class2label, self.class_count, allow_pickle=True)

	def load(self, filename):
		file = np.load(filename, allow_pickle=True)
		self.samples = file['arr_0']
		self.label2class = file['arr_1']
		self.class2label = file['arr_2']
		self.class_count = file['arr_3']