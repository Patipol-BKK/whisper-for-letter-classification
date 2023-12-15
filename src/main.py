# Import modules
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperForAudioClassification, WhisperFeatureExtractor
from trainer import train
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
import numpy as np

# Import supporting scripts
from models import ModelUtils, WhisperForLetterClassification
from configs import WflcConfigs
from audio_dataset import AudioDataset
from trainer import eval_model

import matplotlib.pyplot as plt

# train_1epoch
# print(test_forward)

batch_size = 8
num_samples = 500
# Load Dataset
sloan_letters = ['c', 'd', 'h', 'k', 'n', 'o', 'r', 's', 'v', 'z']

snr_range = [2]

for snr in snr_range:
	full_dataset = AudioDataset(
		'datasets/alphadigit_sloans', 
		'datasets/noise', 
		snr_range = (snr, snr),
		selected_labels = sloan_letters, 
		num_samples = num_samples,
		augment_num=5
	)

	class_weights = full_dataset.class_count/sum(full_dataset.class_count)
	class_weights=torch.tensor(class_weights,dtype=torch.float).cuda()

	train_size = int(0.8 * len(full_dataset))
	val_size = len(full_dataset) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

	wflc_small_config = WflcConfigs.get_config('wflc_small')
	wflc_small = WhisperForLetterClassification(wflc_small_config)
	wflc_small = wflc_small.to(device='cuda:0')

	criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
	optimizer = optim.Adam(wflc_small.parameters(), lr=0.001, weight_decay=0.0001)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	loss = train(wflc_small, train_dataloader, val_dataloader, criterion, optimizer)
	best_weights = loss[4]
	loss = loss[:4]
	print(f'Train Loss: {round(loss[0][-1], 2)}, Train Acc: {round(loss[1][-1], 2)}, Val Loss: {round(loss[2][-1], 2)}, Val Acc: {round(loss[3][-1], 2)}')
	torch.save({
			'model_state_dict': best_weights,
		},f'models/wflc_small_snr{snr}_{num_samples}_aug10_best.npz')

	torch.save({
			'model_state_dict': wflc_small.state_dict(),
		},f'models/wflc_small_snr{snr}_{num_samples}_aug10_last.npz')

	np.save(f'models/wflc_small_snr{snr}_{num_samples}_aug10_losses.npy', np.array(loss, dtype=object), allow_pickle=True)