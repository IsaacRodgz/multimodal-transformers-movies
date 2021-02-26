import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        dim = 312
        self.fc1 = nn.Linear(dim, dim*2)
        self.fc2 = nn.Linear(dim*2, dim*2)
        self.fc3 = nn.Linear(dim*2, args.n_classes)

        # Rectified Linear Unit: ReLU(x) = max(0, x)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(dim*2)
        self.batchnorm2 = nn.BatchNorm1d(dim*2)

    def forward(self, txt, mask, segment, img, poster, metadata):
        x = self.relu(self.fc1(metadata))
        x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        
        return self.fc3(x)