"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn
from pdb import set_trace as st
from misc import config as cfg

class LeNetEncoder(nn.Module):

    def __init__(self, inputc):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(inputc, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            # nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(nn.Linear(50 * 4 * 4, 500),
                   nn.ReLU())        
        # self.fc1 = nn.Sequential(nn.Linear(50 * 5 * 5, 500),
        #            nn.ReLU())        

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(conv_out.size(0), -1))
        return feat


class LeNetEncoder32(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder32, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            # nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )  
        self.fc1 = nn.Sequential(nn.Linear(50 * 5 * 5, 500),
                   nn.ReLU())
                    

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(conv_out.size(0), -1))
        return feat
                 


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self, ncls):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.restored = False
        self.fc2 = nn.Linear(500, ncls)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        # out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(feat)
        return out


