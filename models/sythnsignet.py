"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn
from pdb import set_trace as st


class SythnetEncoder(nn.Module):

    def __init__(self, inputc, nf):
        """Init encoder."""
        super(SythnetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [3 x 32 x 32]
            # output [64 x 15 x 15]
            nn.Conv2d(inputc, 96, 5, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 2nd conv block
            # input [64 x 15 x 15]
            # output [64 x 7 x 7]
            nn.Conv2d(96, 144, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3rd conv block
            # input [64 x 7 x 7]
            # output [128 x 7 x 7]
            nn.Conv2d(144, 256, 5, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc1 = nn.Sequential(nn.Linear(256 * 1 * 1, nf),   #SYN Digits -> SVHN 32*32
           nn.ReLU())

        # self.fc1 = nn.Sequential(nn.Linear(256 * 3 * 3, nf),   #SYN Signs -> GTSRB 48*48
        #    nn.ReLU())

    def forward(self, x):
        """Forward encoder."""
        conv_out = self.encoder(x)
        feat = self.fc1(conv_out.view(conv_out.size(0), -1))
        return feat


class SythnetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self, nf, ncls):
        """Init LeNet encoder."""
        super(SythnetClassifier, self).__init__()
        self.restored = False
        self.fc2 = nn.Linear(nf, ncls)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        # out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(feat)
        return out
