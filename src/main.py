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
num_samples = 2
# Load Dataset
sloan_letters = ['c', 'd', 'h', 'k', 'n', 'o', 'r', 's', 'v', 'z']

snr_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

for snr in snr_range:
	full_dataset = AudioDataset(
		'datasets/alphadigit_sloans', 
		'datasets/noise', 
		snr_range = (0, 0.5),
		selected_labels = sloan_letters, 
		num_samples = num_samples
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
	print(f'Train Loss: {round(loss[0], 2)}, Train Acc: {round(loss[1], 2)}, Val Loss: {round(loss[2], 2)}, Val Acc: {round(loss[3], 2)}')
	torch.save({
			'model_state_dict': wflc_small.state_dict(),
		},f'models/wflc_small_snr{snr}.npz')

	np.save(f'models/wflc_small_snr{snr}_losses.npy', np.array(loss, dtype=object), allow_pickle=True)

# loss = eval_model(wflc_small, dataloader, criterion)
# print(loss)
# # test_forward = wflc_tiny.forward(torch.unsqueeze(full_dataset[0][0], 0))
# test_forward = wflc_small.forward(torch.unsqueeze(full_dataset[0][0], 0))
# # print(test_forward)