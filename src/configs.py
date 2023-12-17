class AudioConfigs:
	configs = {
		'whisper_short_window': {
			'sample_rate': 16000,
			'window_length': 1500 # 1500ms Window
		}
	}

	@classmethod
	def get_config(cls, config_name):
		return cls.configs.get(config_name, None)

class WflcConfigs:
	configs = {
		'wflc_tiny_old': {
			'model_name': 'WhisperForLetterClassification_Tiny',
			'pretrained_whisper': 'openai/whisper-tiny',
			'audio_config': {
				'sample_rate': 16000,
				'window_length': 1500 # 1500ms Window
			},
			'encoder': {
				'module_list': [
					'whisper.encoder.conv1',
					'whisper.encoder.conv2',
					'whisper.encoder.embed_positions',
					'whisper.encoder.layers',
					'whisper.encoder.layer_norm'
				]
			},
			'classifier': {
				'module_list': [
					'whisper.projector', 
					'whisper.classifier'
				]
			}
		},
		'wflc_tiny': {
			'model_name': 'WhisperForLetterClassification_Tiny',
			'pretrained_whisper': 'openai/whisper-tiny',
			'audio_config': {
				'sample_rate': 16000,
				'window_length': 1500 # 1500ms Window
			},
			'encoder': {
				'module_list': [
					'whisper.encoder'
				]
			},
			'projector': {
				'module_list': [
					'whisper.projector'
				]
			},
			'classifier': {
				'module_list': [
					{
						'name': 'Linear',
						'kwargs': {
							'in_features': 256,
							'out_features': 11,
							'bias': True
						}
					}
				]
			}
		},
		'wflc_small': {
			'model_name': 'WhisperForLetterClassification_Small',
			'pretrained_whisper': 'openai/whisper-small',
			'audio_config': {
				'sample_rate': 16000,
				'window_length': 1500 # 1500ms Window
			},
			'encoder': {
				'module_list': [
					'whisper.encoder'
				]
			},
			'projector': {
				'module_list': [
					'whisper.projector'
				]
			},
			'classifier': {
				'module_list': [
					{
						'name': 'Linear',
						'kwargs': {
							'in_features': 256,
							'out_features': 11,
							'bias': True
						}
					}
				]
			}
		},
		'wflc_base': {
			'model_name': 'WhisperForLetterClassification_Base',
			'pretrained_whisper': 'openai/whisper-base',
			'audio_config': {
				'sample_rate': 16000,
				'window_length': 1500 # 1500ms Window
			},
			'encoder': {
				'module_list': [
					'whisper.encoder'
				]
			},
			'projector': {
				'module_list': [
					'whisper.projector'
				]
			},
			'classifier': {
				'module_list': [
					{
						'name': 'Linear',
						'kwargs': {
							'in_features': 256,
							'out_features': 11,
							'bias': True
						}
					}
				]
			}
		}
	}

	@classmethod
	def get_config(cls, config_name):
		return cls.configs.get(config_name, None)