import torch
from torch import nn


class ScalarProjHead(nn.Module):
    def __init__(self, latent_dim, proj_dim):
        super().__init__()
        self.layer1 = nn.Linear(latent_dim, intermediate_dim)
        self.layer2 = nn.Linear(intermediate_dim, proj_dim)
        self.layer3 = nn.Linear(proj_dim, 1)
        self.act = nn.ReLU()
        self.final_act = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(self.act(x))
        x = self.layer3(self.act(x))
        
        return self.final_act(x)




