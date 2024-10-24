import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_class = 26):
        super(MLP,self).__init__()
        if n_class == 26:
            self.dim = 28 * 28
            hidden_1 = 512
            hidden_2 = 512
            self.dropout = nn.Dropout(0.2)
        # number of hidden nodes in each layer (512)
        elif n_class == 50:
            self.dim = 28 * 28 * 5
            hidden_1 = 512
            hidden_2 = 512
            self.dropout = nn.Dropout(0.5)
        # linear layer (dim -> hidden_1)
        self.fc1 = nn.Linear(self.dim, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> n_class)
        self.fc3 = nn.Linear(hidden_2, n_class)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1, self.dim)
        # add hidden layer, with relu activation function
        x = self.dropout(F.relu(self.fc1(x)))
         # add hidden layer, with relu activation function
        x = self.dropout(F.relu(self.fc2(x)))
        # add output layer
        x = self.fc3(x)
        return x