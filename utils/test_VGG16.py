import torch
from tools.data_tools import device

DEVICE = device()
def test(model, test_loader):
  model.eval()
  n_correct = 0
  n_samples = 0
  for inputs, labels in test_loader:
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

  acc = 100.0 * n_correct / n_samples
  print(f'Accuracy of the network on the 10000 test images: {acc} %')
  return acc

