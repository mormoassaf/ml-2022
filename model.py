from torch import nn
import torch
import torch.nn.functional as F

# References used:
# https://www.kaggle.com/code/robinreni/signature-classification-using-siamese-pytorch
# https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        # Defining the fully connected layers using convolutional layers of N neurons
        self.fc1 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1),
            nn.Flatten(),
        )

    
    # Retrieve the distance between the 2 embeddings
    def compare_embeddings(self, embedding1, embedding2):
        return F.pairwise_distance(embedding1, embedding2)

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
        # output = torch.concat((output1, output2), 1)
        # output = self.fc2(output).squeeze()
        # return output

    def predict(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        # similarity between the 2 outputs
        return self.compare_embeddings(output1, output2)
