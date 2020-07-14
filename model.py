import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Model(nn.Module):
    def __init__(self, n_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.dropout(x)
        hidden = x
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        x = self.dropout(x)
        return x
