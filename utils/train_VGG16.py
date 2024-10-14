import torch
from tools.plots import loss_curve
from tools.data_tools import device

DEVICE = device()

def train(model, train_loader, optimizer, criterion, num_epochs):
  model = model.to(DEVICE)
  model.train()

  loss_result = []
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)
      outputs = model(images)

      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i%100 == 0:
        print(f"epoch: {epoch + 1}/{num_epochs}, step: {i + 1}/{len(train_loader)}, loss: {loss.item()}")
        loss_result.append(loss.item())
  loss_curve(loss_result)
  return model
