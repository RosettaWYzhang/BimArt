import torch.nn as nn

    
class MLP_Encoder(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, input_dim, latent_dim, layer1_dim=1024, layer2_dim=1024):
    super().__init__()
    self.input_dim = input_dim,
    self.layers = nn.Sequential(
      nn.Linear(input_dim, layer1_dim),
      nn.ReLU(),
      nn.Linear(layer1_dim, layer2_dim),
      nn.ReLU(),
      nn.Linear(layer2_dim, latent_dim)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
