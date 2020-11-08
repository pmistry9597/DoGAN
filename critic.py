# critic network, outputs a score that should be higher for real images, lower for fake images
# it will try to make this loss function bigger: D(x) - D(G) where D(x) is avg of

from torch import nn

# create convolutional block
def createConv(inChannels, outChannels, kernel_size, batchNorm=False):
    conv = nn.Conv2d(inChannels, outChannels, kernel_size=(kernel_size, kernel_size))
    maxpool = nn.MaxPool2d(2,2)

    if batchNorm:
        return nn.Sequential( conv, nn.BatchNorm2d(outChannels), nn.LeakyReLU(), maxpool )

    return nn.Sequential( conv, nn.LeakyReLU(), maxpool )

#create fully connected layer
def createFC(inNum, outNum, batchNorm=False):
    return nn.Sequential( nn.Linear(inNum, outNum), nn.BatchNorm1d(outNum), nn.LeakyReLU() )

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        #convolutional layers
        self.convBlocks = nn.Sequential( 
            createConv(3, 32, kernel_size=5, batchNorm=True), 
            createConv(32, 64, kernel_size=5, batchNorm=True),
            createConv(64, 128, kernel_size=5, batchNorm=True)
        )

        #fully connected layers
        self.fcBlocks = nn.Sequential(
            createFC(10368, 100, batchNorm=True),
            createFC(100, 50, batchNorm=True),
            nn.Linear(50, 1)
        )

        self.batchSize = 1

    def forward(self, x):
        x = self.convBlocks(x)
        x = x.view([self.batchSize, -1])
        x = self.fcBlocks(x)
        return x

critic = Critic()