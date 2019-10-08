"""Discriminator model for ADDA."""

from torch import nn
from pdb import set_trace as st


class Discriminator_feat(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims, gpu_ids = [0,1,2,3]):
        """Init discriminator."""
        super(Discriminator_feat, self).__init__()

        self.restored = False
        
        self.gpu_ids = gpu_ids

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class Discriminator_img(nn.Module):
    def __init__(self, nc=1, ndf=64, gpu_ids = [0,1,2,3]):
        super(Discriminator_img,self).__init__()
        self.restored = False
        self.gpu_ids = gpu_ids
        
        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),

        )
        
    def forward(self, input):
        output = self.model(input)
        return output
        



