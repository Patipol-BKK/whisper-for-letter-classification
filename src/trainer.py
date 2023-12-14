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
	best_loss = 1000000
	best_weights = None

	train_losses = []
	train_accuracies = []

	val_losses = []
	val_accuracies = []
	pbar = tqdm(total=epochs, desc='Training')
	for epoch in range(epochs):
		epoch_loss, epoch_accuracy = train_1epoch(model, train_dataloader, criterion, optimizer, device)
		train_losses.append(epoch_loss)
		train_accuracies.append(epoch_accuracy)

		epoch_loss, epoch_accuracy = eval_model(model, val_dataloader, criterion, device)
		val_losses.append(epoch_loss)
		val_accuracies.append(epoch_accuracy)

		pbar.set_postfix({
			'Train Loss': train_losses[-1], 
			'Train Acc': train_accuracies[-1],
			'Val Loss': val_losses[-1],
			'Val Acc': val_accuracies[-1]
		})

		if epoch_loss < best_loss:
			best_loss = epoch_loss
			print(f'Best Loss: {best_loss}, Acc: {epoch_accuracy}')
			best_weights = model.state_dict()

		pbar.update(1)

	return train_losses, train_accuracies, val_losses, val_accuracies, best_weights
# def train_cv