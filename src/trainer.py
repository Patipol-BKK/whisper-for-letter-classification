import torch
from tqdm import tqdm
from models import WhisperForLetterClassification

def compute_corrects(outputs, labels):
	# Convert model outputs to predicted class labels
	_, predicted = torch.max(outputs, dim=1)
	
	# Calculate the number of correct predictions
	correct = (predicted == labels).sum().item()
	
	return correct

def train_1epoch(model, dataloader, criterion, optimizer, device='cuda:0'):
	model.train()
	avg_loss = 0
	corrects = 0
	num_items = 0

	for data in dataloader:
		inputs, labels = data

		inputs = inputs.to(device=device)
		labels = labels.to(device=device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		num_items += labels.size()[0]
		avg_loss += loss.item()
		corrects += compute_corrects(outputs, labels)

	avg_loss /= num_items
	accuracy = corrects / num_items

	return avg_loss, accuracy


def eval_model(model, dataloader, criterion, device='cuda:0'):
	model.eval()
	avg_loss = 0
	corrects = 0
	num_items = 0
	with torch.no_grad():
		for data in dataloader:
			inputs, labels = data

			inputs = inputs.to(device=device)
			labels = labels.to(device=device)

			outputs = model(inputs)
			loss = criterion(outputs, labels)

			num_items += labels.size()[0]
			avg_loss += loss.item()
			corrects += compute_corrects(outputs, labels)

	avg_loss /= num_items
	accuracy = corrects / num_items
	return avg_loss, accuracy

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device='cuda:0', epochs=100):
	train_losses = []
	train_accuracies = []

	val_losses = []
	val_accuracies = []
	for epoch in tqdm(range(epochs), desc='Training Model'):
		epoch_loss, epoch_accuracy = train_1epoch(model, train_dataloader, criterion, optimizer, device)
		train_losses.append(epoch_loss)
		train_accuracies.append(epoch_accuracy)

		epoch_loss, epoch_accuracy = eval_model(model, val_dataloader, criterion, device)
		val_losses.append(epoch_loss)
		val_accuracies.append(epoch_accuracy)

	return train_losses, train_accuracies, val_losses, val_accuracies
# def train_cv