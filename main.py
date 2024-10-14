import torch.optim as optim
from utils.data import dataloader
from utils.train_VGG16 import train
from utils.test_VGG16 import test
from models.VGG16 import VGGNet

def main(num_epochs = 10):
  train_loader, test_loader = dataloader()
  model_VGGNet = VGGNet(img_size = 224, num_classes = 10)
  optimizer = optim.Adam(model_VGGNet.parameters(), lr = 0.0001)
  criterion = nn.CrossEntropyLoss()
  model = train(model_VGG, train_loader, optimizer, criterion, num_epochs)
  
  test_result = test(model, test_loader)


if __name__ == "__main__":
  main()


  
