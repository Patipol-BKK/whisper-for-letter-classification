# Import modules
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperForAudioClassification, WhisperFeatureExtractor
import copy

# Import supporting scripts
from models import ModelUtils, WhisperForLetterClassification
from configs import WflcConfigs

whisper = WhisperForAudioClassification.from_pretrained('openai/whisper-tiny')
# print(ModelUtils.model_module(whisper, 'encoder.conv1'))
whisper = WhisperModel.from_pretrained('openai/whisper-tiny')

wflc_tiny_config = WflcConfigs.get_config('wflc_tiny')
wflc_tiny = WhisperForLetterClassification(wflc_tiny_config)

wflc_tiny.forward(torch.randn(1, 80, 150))

torch.save({
		'model_state_dict': wflc_tiny.state_dict(),
	},'wflc_tiny.npz')