"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn
from pdb import set_trace as st


class LeNetGenerator(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, input_dims, outputc, gpu_ids = [0,1,2,3]):
        """Init LeNet encoder."""
        super(LeNetGenerator, self).__init__()
        self.restored = False
        self.gpu_ids = gpu_ids

        model = [nn.ConvTranspose2d(input_dims, 256, kernel_size=5, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True)]
        model += [nn.ConvTranspose2d(256, 128, kernel_size=5, stride=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True)]
        model += [nn.ConvTranspose2d(128, outputc, kernel_size=2, stride=2, padding=1, bias=False),
                nn.Tanh()]

        self.decoder = nn.Sequential(*model)
                    
    def forward(self, input):
        # """Forward the LeNet."""
        out = self.decoder(input)
        return out



class SythnetGenerator(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, input_dims, outputc, gpu_ids = [0,1,2,3]):
        """Init LeNet encoder."""
        super(SythnetGenerator, self).__init__()
        self.restored = False
        self.gpu_ids = gpu_ids

        model = [nn.ConvTranspose2d(input_dims, 512, kernel_size=4, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True)]
        model += [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True)]        
        model += [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True)]
        model += [nn.ConvTranspose2d(128, outputc, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()]


        self.decoder = nn.Sequential(*model)
                    
    def forward(self, input):
        # """Forward the LeNet."""
        out = self.decoder(input)
        return out


