import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import numpy as np
import loader

class DataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images
        self.transform = transforms.Compose([
            transforms.Resize((256, 256, 64)),
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
    metadatafp = 'image_metadata.csv'
    df = pd.read_csv(metadatafp)
    df = df[df['patient'] == 'Breast_MRI_001']
    data_dir = '../cs235-data/'
    # data_dir = '/home/tomasbencomo/final-project/data'
    scans = loader.load_cases(df, data_dir, 1)
    er_labels = df['ER']
    pr_labels = df['PR']
    her2_labels = df['HER2']
    print("Completed loading!")

    X_train, X_test, y_train, y_test = train_test_split(scans, er_labels, test_size=.2, random_state=42)

    train_dataset = DataSet(X_train, y_train)
    val_dataset = DataSet(X_test, y_test)

    train_gen = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_gen = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True)

    train(net, train_gen, val_gen)

if __name__ == '__main__':
    main()
