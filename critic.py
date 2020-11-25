# critic network, outputs a score that should be higher for real images, lower for fake images
# it will try to make this loss function bigger: D(x) - D(G) where D(x) is score of real images, D(G) is score of generated images

from torch import nn

# create convolutional block
def createConv(inChannels, outChannels, kernel_size, stride, padding=0, batchNorm=False):
    conv = nn.Conv2d(inChannels, outChannels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding))
    #maxpool = nn.MaxPool2d(2,2)

    if batchNorm:
        return nn.Sequential( conv, nn.BatchNorm2d(outChannels), nn.LeakyReLU() )

    return nn.Sequential( conv, nn.LeakyReLU(0.2) )

#create fully connected layer
def createFC(inNum, outNum, batchNorm=False):
    if batchNorm:
        return nn.Sequential( nn.Linear(inNum, outNum), nn.BatchNorm1d(outNum), nn.LeakyReLU() )
    
    return nn.Sequential( nn.Linear(inNum, outNum), nn.LeakyReLU(0.2) )

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        #convolutional layers
        self.convBlocks = nn.Sequential( 
            createConv(3, 32, kernel_size=3, stride=2, padding=1, batchNorm=False), 
            createConv(32, 64, kernel_size=3, stride=2, padding=1, batchNorm=False),
            createConv(64, 128, kernel_size=5, stride=1, padding=0, batchNorm=False),
            createConv(128, 256, kernel_size=5, stride=1, padding=0, batchNorm=False)
            #createConv(128, 256, kernel_size=5, stride=1)# batchNorm=True)
        )

        #fully connected layers
        self.fcBlocks = nn.Sequential(
            nn.Linear(73984, 1)
        )

        self.batchSize = 1

    def forward(self, x):
        x = self.convBlocks(x)
        x = x.view([self.batchSize, -1])
        x = self.fcBlocks(x)
        return x