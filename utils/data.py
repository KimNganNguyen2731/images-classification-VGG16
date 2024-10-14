from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def dataloader(batch_size: int = 64):
  # Transform for dataset
  transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  # Data loading
  train_dataset = datasets.CIFAR10(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform)
  test_dataset = datasets.CIFAR10(root='./data',
                                   train=False,
                                   download=True,
                                   transform=transform)
  # Data Loader
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size, 
                            shuffle=True)
  test_loader = DataLoader(test_dataset,
                           batch_size = batch_size,
                           shuffle = False)
  return train_loader, test_loader
