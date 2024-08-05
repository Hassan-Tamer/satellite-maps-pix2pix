import torch
import torch.nn as nn

class ConvGenerator_Block(nn.Module):
    '''
    This class defines the convolutional Generator_Block used in the discriminator
    '''
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, BN=True,dropout=0,padding=1):
        super().__init__()
        layers = []

        layers.append(nn.ReflectionPad2d(padding))
        padding=0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding, bias=False)

        layers.append(self.conv)

        if BN:
            bn = nn.BatchNorm2d(out_channels)
            layers.append(bn)

        layers.append(nn.LeakyReLU(0.2))
        
        if dropout>0:
            dropout = nn.Dropout(dropout)
            layers.append(dropout)
    
        self.conv_Generator_Block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_Generator_Block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # in_channels is multiplied by 2 because we are concatenating the input image with the ground truth image
        layers = [ConvGenerator_Block(in_channels*2, features[0],BN = False)]
        in_channels = features[0]
        for feature in features[1:]:
            stride = 2
            if feature == features[-1]:
                stride = 1
            else: stride = 2

            layers.append(ConvGenerator_Block(in_channels, feature, kernel_size=4, stride=stride, BN=True))
            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.model(x)
        return x
    
class Generator_Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2) if down else nn.ReLU(),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(use_dropout)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Generator_Block(64,128)
        self.down2 = Generator_Block(128, 256)
        self.down3 = Generator_Block(256,512)
        self.down4 = Generator_Block(512, 512)
        self.down5 = Generator_Block(512, 512)
        self.down6 = Generator_Block(512, 512)
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())

        self.up1 = Generator_Block(512, 512, down=False,use_dropout=True)
        self.up2 = Generator_Block(1024, 512, down=False,use_dropout=True)
        self.up3 = Generator_Block(1024,512, down=False,use_dropout=True)
        self.up4 = Generator_Block(1024, 512, down=False)
        self.up5 = Generator_Block(1024, 256, down=False)
        self.up6 = Generator_Block(512, 128, down=False)
        self.up7 = Generator_Block(256, 64, down=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))
