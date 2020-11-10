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
        self.ct1 = createConvT(256, 128, kernel_size=4)
        self.ct2 = createConvT(128, 128, kernel_size=4)
        self.ct3 = createConvT(128, 64, kernel_size=4, stride=2)
        self.ct4 = createConvT(64, 32, kernel_size=3, stride=2)
        #self.ct5 = createConvT(32, 3, kernel_size=8, stride=2)
        self.ct5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=(2,2)),
            nn.Tanh()
        )

        nz = 100
        self.nz = nz
        self.main = nn.Sequential(
            # nz will be the input to the first convolution
            nn.ConvTranspose2d(
                nz, 512, kernel_size=4, 
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, 
                stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=5, 
                stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=5, 
                stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 3, kernel_size=4, 
                stride=2, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        '''x = self.ct0(x)
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        x = self.ct4(x)
        x = self.ct5(x)'''
        x = self.main(x)
        return x
'''
import torch
rando = torch.randn([1,100,1,1])
gen = Generator()
gened = gen(rando)
print(gened.shape)
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(gened[0].permute(1,2,0).detach().numpy())

plt.figure()
rando = torch.randn([1,100,1,1])
gened = gen(rando)
plt.imshow(gened[0].permute(1,2,0).detach().numpy())
plt.show()
'''