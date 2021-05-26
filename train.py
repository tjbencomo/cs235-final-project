import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report
from torchvision import transforms
import torch
import numpy as np
import loader

class DataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

def train(net, training_generator, val_generator, learning_rate = 0.0001, num_epochs = 5):
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)

    for i in range(num_epochs):
        net.train()
        losses = []
        for (image,label) in training_generator:
            optimizer.zero_grad()
            image = image.type(torch.FloatTensor)
            out = net(image.cuda())
            loss = loss_func(out, label.cuda())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy())
      #print('\r Epoch {} Step {}/{} Loss {:.2f'}.format(i+1, j, int(len(training_generator)) ), np.mean(losses), end = "")
  
    net.eval()
    accuracy = []
    for j, (image,label) in enumerate(val_generator):
        image = image.type(torch.FloatTensor)
        out = net(image.cuda())
        best = np.argmax(out.data.cpu().numpy(), axis =-1)
        accuracy.extend(list(best == label.data.cpu().numpy()))
    print('\n Accuracy is ', str(np.mean(accuracy) * 100))

def main():
  train(net, training_generator, val_generator)

if __name__ == '__main__':
    main()
