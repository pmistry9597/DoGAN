# generator network, output images based on input random vector
# try to maximize this loss function: D(G(z))

from torch import nn

#create convolutional transpose block with leaky relu activation and batch normalization
def createConvT(inChannels, outChannels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(inChannels, outChannels, kernel_size, stride=(stride,stride)),
        nn.BatchNorm2d(outChannels),
        nn.LeakyReLU()
    )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ct0 = createConvT(512, 256, kernel_size=5)
        self.ct1 = createConvT(256, 128, kernel_size=6)
        self.ct2 = createConvT(128, 128, kernel_size=6)
        self.ct3 = createConvT(128, 64, kernel_size=7)
        self.ct4 = createConvT(64, 32, kernel_size=7, stride=2)
        self.ct5 = createConvT(32, 3, kernel_size=8, stride=2)

    def forward(self, x):
        x = self.ct0(x)
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        x = self.ct4(x)
        x = self.ct5(x)
        return x