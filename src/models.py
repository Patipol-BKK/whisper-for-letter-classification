import torch.nn as nn
import copy
from transformers import WhisperForAudioClassification

class ModelUtils:
	# Creates a torch.nn module from the specified module name, args, and kwargs
	@classmethod
	def torchnn_module(cls, config):
		module_name = config.get('name')
		module_args = config.get('args', [])
		module_kwargs = config.get('kwargs', {})

		module_class = getattr(nn, module_name)
		module_instance = module_class(*module_args, **module_kwargs)

		return module_instance

	# Creates a deep copy of the specified module from the original model
	@classmethod
	def model_module(cls, source_model, module_name):
		# Get module list
		modules = source_model.named_modules()

		# Iterate through list until specified module name is matched
		for module in modules:
			if module[0] == module_name:
				return copy.deepcopy(module[1])
		

class WhisperForLetterClassification(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.audio_config = config['audio_config']

		self.pretrained_whisper = config['pretrained_whisper']
		whisper = WhisperForAudioClassification.from_pretrained(config['pretrained_whisper'])
		self.encoder = nn.Sequential(*self._parse_modules(whisper, config['encoder']['module_list']))
		self.projector = nn.Sequential(*self._parse_modules(whisper, config['projector']['module_list']))
		self.classifier = nn.Sequential(*self._parse_modules(whisper, config['classifier']['module_list']))
		for module in self.encoder.named_modules():
			module_name = module[0]

			if 'embed_positions' in module_name:
				in_channels = int(config['audio_config']['window_length'] / 20)
				out_channels = module[1].embedding_dim

				original_embedding_weights = module[1].weight.data
				new_embedding = nn.Embedding(in_channels, out_channels)
				new_embedding.weight.data.copy_(original_embedding_weights[:in_channels, :])
				module = new_embedding

		for param in self.encoder.parameters():
			param.requires_grad = False


	def _parse_modules(self, whisper, modules):
		module_list = []
		for module in modules:
			if isinstance(module, dict):
				module_list.append(ModelUtils.torchnn_module(module))
			elif module.split('.')[0] == 'whisper':
				whisper_module = '.'.join(module.split('.')[1:])
				whisper_module_instance = ModelUtils.model_module(whisper, whisper_module)

				# Replace original positional encoder according input length in config
				if whisper_module == 'encoder':
					original_embedding_weights = whisper_module_instance.embed_positions.weight.data

					in_channels = int(self.audio_config['window_length'] / 20)
					out_channels = whisper_module_instance.embed_positions.embedding_dim

					new_embedding = nn.Embedding(in_channels, out_channels)
					new_embedding.weight.data.copy_(original_embedding_weights[:in_channels, :])
					whisper_module_instance.embed_positions = new_embedding

				module_list.append(whisper_module_instance)
				
		return module_list


	def forward(self, x):
		output = self.encoder(x)['last_hidden_state']
		output = self.projector(output)
		pooled_output = output.mean(dim=1)

		pooled_output = self.classifier(pooled_output)
		return pooled_output