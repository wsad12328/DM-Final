import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        
        out1 = self.drop(self.relu(self.bn1(self.fc1(x))))
        out2 = self.drop(self.relu(self.bn2(self.fc2(out1))))
        out2 = out2 + out1  # Residual

        out3 = self.drop(self.relu(self.bn3(self.fc3(out2))))
        out3 = out3 + out2  # Residual

        out4 = self.drop(self.relu(self.bn4(self.fc4(out3))))
        z = out4 + out3  # Residual

        return z

class ProjectionHead(nn.Module):  # g
    def __init__(self, input_dim, projection_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        z = self.net(x) + x
        return F.normalize(z, dim=1) 

class ClassifierHead(nn.Module):  # h
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)