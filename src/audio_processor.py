import sounddevice as sd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import threading

import ml_collections

# Audio Processor Config

class AudioProcessor:
	def __init__(self, config, callback):
		self.sample_rate = '16000'
		self.callback = callback
	def start