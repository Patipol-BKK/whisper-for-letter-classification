# Import modules
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperForAudioClassification, WhisperFeatureExtractor
import copy

# Import supporting scripts
from models import ModelUtils, WhisperForLetterClassification
from configs import WflcConfigs

# whisper = WhisperForAudioClassification.from_pretrained('openai/whisper-tiny')
# print(ModelUtils.model_layer(whisper, 'encoder.conv1'))
# whisper = WhisperModel.from_pretrained('openai/whisper-tiny')
# print(whisper)

# config = WflcConfigs.get_config('wflc_tiny')
# model = WhisperForLetterClassification(config)
# print(model)
# print(model(torch.randn(1, 80, 3000).type(torch.FloatTensor)))

# import datetime


import numpy as np
import sounddevice as sd
import soundfile as sf

duration = 1.5  # Duration of each chunk in seconds
overlap = 0.5  # Overlap duration in seconds
sample_rate = 16000  # Sample rate in Hz

chunk_samples = int(duration * sample_rate)
overlap_samples = int(overlap * sample_rate)

recorded_audio = np.array([])

def stream_callback(indata, frames, time, status):
	global recorded_audio
	recorded_audio = np.append(recorded_audio, indata)

import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import warnings


def inference_thread():
	global out
	global recorded_audio
	feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')

	CUT_OFF = 150

	model = WhisperForAudioClassification.from_pretrained('openai/whisper-small')
	model.classifier = nn.Linear(in_features=256, out_features=11, bias=True)

	original_embedding_weights = model.encoder.embed_positions.weight.data
	new_embedding = nn.Embedding(int(CUT_OFF/2), 768)
	new_embedding.weight.data.copy_(original_embedding_weights[:int(CUT_OFF/2), :])
	model.encoder.embed_positions = new_embedding

	model.load_state_dict(torch.load('whisper_new/whisper_sloan_small_5k_best_acc.npz'))
	# model.load_state_dict(torch.load('whisper_new/whisper_freeze_small_10k_best_acc.npz'))
	model = model
	model.eval()
	print('asdasd')
	try:
		with sd.InputStream(callback=stream_callback, channels=1, samplerate=sample_rate):
			while True:
				# Record a chunk of audio
				sd.sleep(100)

				# Check if the recorded audio is longer than the desired duration
				if len(recorded_audio) >= chunk_samples:
					# Get the last chunk_samples samples from the recorded audio
					chunk = recorded_audio[len(recorded_audio)-chunk_samples:]

					# Filter out quiet periods
					if np.max(chunk) > 0.027:
						features = feature_extractor(chunk, sample_rate=sample_rate)['input_features'][0][:, :CUT_OFF]
						inputs = torch.unsqueeze(torch.tensor(features), 0)
						with warnings.catch_warnings():
							warnings.simplefilter('ignore')
							model_out = nn.functional.relu(model.forward(inputs)[0]).cpu().detach()
						if np.max(chunk) > 0.007:
							out = model_out
							class_id = torch.argmax(out)
							print(class_id, out[0], np.max(chunk))
							
						# Process the chunk (e.g., save to file, perform analysis, etc.)
						# Here, we'll just print the shape of the chunk
						

						# Remove the processed chunk from the recorded audio
						recorded_audio = recorded_audio[len(recorded_audio)-chunk_samples + overlap_samples:]
					else:
						out = torch.zeros((1, 11))
					
	except KeyboardInterrupt:
		return
import random
import plotly.graph_objects as go
import time

out = torch.zeros((1, 11))

current_idx = 0
sloans = random.sample(['C', 'D', 'H', 'K', 'N', 'O', 'R', 'S', 'V', 'Z'], 5)
corrects = []
prev_ans = time.time()

def create_figure():
	global out
	global sloans
	global corrects
	global current_idx
	global prev_ans

	if current_idx >= 5:
		current_idx = 0
		sloans = random.sample(['C', 'D', 'H', 'K', 'N', 'O', 'R', 'S', 'V', 'Z'], 5)
		corrects = []

	numbers = out[0].numpy()
	class_id = torch.argmax(out)
	colors = ['blue' for value in numbers]
	if numbers[class_id] > 12 and time.time() - prev_ans > 2:
		colors[class_id] = 'red'
		if ['C', 'D', 'H', 'K', 'N', 'O', 'R', 'S', 'V', 'Z', 'Other'][class_id] == sloans[current_idx] and class_id != 10:
			corrects.append('✓')
			current_idx += 1
			prev_ans = time.time()
		elif ['C', 'D', 'H', 'K', 'N', 'O', 'R', 'S', 'V', 'Z', 'Other'][class_id] != sloans[current_idx] and class_id != 10:
			corrects.append('X')
			current_idx += 1
			prev_ans = time.time()

	fig = go.Figure(data=go.Bar(
		x=['C', 'D', 'H', 'K', 'N', 'O', 'R', 'S', 'V', 'Z', 'Other'], 
		y=numbers,
		marker=dict(color=colors)), layout_yaxis_range=[0,20])
	return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
	html.Div([
		html.H2(id='live-update-text'),
		dcc.Graph(
			id='bar-chart',
			figure=create_figure()
		),
		dcc.Interval(
			id='interval-component',
			interval=1*1000, # in milliseconds
			n_intervals=0
		),
		html.H3(id='sloan-words')
	])
)


import sys

@callback(Output('live-update-text', 'children'),
			  Input('interval-component', 'n_intervals'))

def update_metrics(n):
	global out
	numbers = out[0].numpy()
	class_id = torch.argmax(out)
	classes = ['C', 'D', 'H', 'K', 'N','O', 'R', 'S', 'V', 'Z', 'Other']
	style = {'padding': '16px', 'fontSize': '50px'}
	if numbers[class_id] > 12:
		return [
			html.Span(f'Prediction: {classes[class_id]}', style=style)
		]
	else:
		return [
			html.Span(f'Prediction: None', style=style)
		]



@callback(Output('sloan-words', 'children'),
			  [Input('interval-component', 'n_intervals')])
def update_sloan(n):
	style = {'padding': '16px', 'fontSize': '50px'}
	return [
		html.Span(f'{" ".join(sloans)}', style=style),
		html.Span(f'{" ".join(corrects)}', style=style)
	]
	

@callback(Output('bar-chart', 'figure'),
			  [Input('interval-component', 'n_intervals')])
def update_figure(n):
	return create_figure()

from threading import Thread


if __name__ == '__main__':
	inference = Thread(target=inference_thread)
	inference.start()
	app.run(debug=False) 
	# Start recording audio
	
