from ccbdl.network.base import BaseNetwork
from ccbdl.utils.logging import get_logger
from ccbdl.network.nlrl import NLRL_AO, InverseSigmoid
import torch


class CGAN(BaseNetwork):
    def __init__(self, name: str, noise_dim: int, hidden_channels: int, final_layer:str, debug=False):
        super().__init__(name, debug)
        self.generator = Generator(noise_dim, hidden_channels, 10)
        self.discriminator = Discriminator(hidden_channels, 10, final_layer)

    def forward(self, x, labels):
        generated_images = self.generator(x, labels)
        discriminated_images = self.discriminator(generated_images, labels)
        return generated_images, discriminated_images
    

class Generator(torch.nn.Module):
    def __init__(self, noise_dim: int, hidden_channels: int, num_classes: int):
        super(Generator, self).__init__()
        self.embedding = torch.nn.Embedding(num_classes, num_classes)
        initial_hidden_channels = hidden_channels*8
        
        self.gen = torch.nn.Sequential(
            torch.nn.Linear(noise_dim + num_classes, initial_hidden_channels*4*4),
            torch.nn.ReLU(),

            torch.nn.Unflatten(1, (initial_hidden_channels, 4, 4)),

            torch.nn.ConvTranspose2d(initial_hidden_channels, hidden_channels*4, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),

            torch.nn.Conv2d(hidden_channels, 1, 3, 1, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, noise, labels):
        labels = self.embedding(labels)
        x = torch.cat((noise, labels), -1)
        return self.gen(x)


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_classes: int, final_layer:str):
        super(Discriminator, self).__init__()
        self.embedding = torch.nn.Embedding(num_classes, num_classes)
        
        self.dis = torch.nn.Sequential()
        in_channels = 1 + num_classes # Due to greyscale data
        initial_out_channels = hidden_channels//2

        for idx in range(3):
            if idx % 3 == 0:
                out_channels = int(initial_out_channels * 2)
                initial_out_channels *= 2
            self.dis.append(GANConvBlock(in_channels,
                                        out_channels,
                                        5 if idx == 0 else 3,
                                        0 if idx == 0 else 1,
                                        ))
            if idx % 4 ==0:
                self.dis.append(torch.nn.Dropout2d(p=0.2))
            if idx == 3 // 2:
                self.dis.append(torch.nn.MaxPool2d(2))
            in_channels = out_channels

        self.dis.append(GANConvBlock(in_channels, 64, 3, 0))
        self.dis.append(GANConvBlock(64, 32, 3, 0))
        self.dis.append(GANConvBlock(32, 16, 3, 0))
        self.dis.append(torch.nn.AdaptiveMaxPool2d(4))
        self.dis.append(torch.nn.Conv2d(16, 8, 4))
        self.dis.append(torch.nn.Flatten())       
        if final_layer.lower() == 'linear':
            self.dis.append(torch.nn.Linear(8, 1))
            self.dis.append(torch.nn.Sigmoid())
        elif final_layer.lower() == 'nlrl':
            self.dis.append(torch.nn.Sigmoid())
            self.dis.append(NLRL_AO(8, 1))
        else:
            raise ValueError(
                f"Invalid value for final_layer: {final_layer}, it should be 'linear', or 'nlrl'")

    def forward(self, x, labels):
        labels = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, labels), 1)
        return self.dis(x)


class GANConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GANConvBlock, self).__init__()
        self.sequence = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                                            torch.nn.BatchNorm2d(out_channels),
                                            torch.nn.LeakyReLU(0.2, inplace=True))
    def forward(self, ins):
        return self.sequence(ins)
    

class GAN(BaseNetwork):
    def __init__(self, name: str, noise_dim: int, hidden_channels: int, final_layer: str, debug=False):
        super().__init__(name, debug)
        initial_hidden_channels = hidden_channels*8

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, initial_hidden_channels*4*4),
            torch.nn.ReLU(),

            torch.nn.Unflatten(1, (initial_hidden_channels, 4, 4)),

            torch.nn.ConvTranspose2d(initial_hidden_channels, hidden_channels*4, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),

            torch.nn.Conv2d(hidden_channels, 1, 3, 1, 1, bias=False),
            torch.nn.Sigmoid()
        )

        self.discriminator = torch.nn.Sequential()
        in_channels = 1 # Due to greyscale data
        initial_out_channels = hidden_channels//2

        for idx in range(3):
            if idx % 3 == 0:
                out_channels = int(initial_out_channels * 2)
                initial_out_channels *= 2
            self.discriminator.append(GANConvBlock(in_channels,
                                        out_channels,
                                        5 if idx == 0 else 3,
                                        0 if idx == 0 else 1,
                                        ))
            if idx % 4 ==0:
                self.discriminator.append(torch.nn.Dropout2d(p=0.2))
            if idx == 3 // 2:
                self.discriminator.append(torch.nn.MaxPool2d(2))
            in_channels = out_channels

        self.discriminator.append(GANConvBlock(in_channels, 64, 3, 0))
        self.discriminator.append(GANConvBlock(64, 32, 3, 0))
        self.discriminator.append(GANConvBlock(32, 16, 3, 0))
        self.discriminator.append(torch.nn.AdaptiveMaxPool2d(4))
        self.discriminator.append(torch.nn.Conv2d(16, 8, 4))
        self.discriminator.append(torch.nn.Flatten())       
        if final_layer.lower() == 'linear':
            self.discriminator.append(torch.nn.Linear(8, 1))
            self.discriminator.append(torch.nn.Sigmoid())
        elif final_layer.lower() == 'nlrl':
            self.discriminator.append(torch.nn.Sigmoid())
            self.discriminator.append(NLRL_AO(8, 1))
        else:
            raise ValueError(
                f"Invalid value for final_layer: {final_layer}, it should be 'linear', or 'nlrl'")

    def forward(self, x):
        generated_images = self.generator(x)
        discriminated_images = self.discriminator(generated_images)
        return generated_images, discriminated_images
    

class CNN(BaseNetwork):
    def __init__(self,
                 in_channels: int,
                 name: str,
                 initial_out_channels: int, 
                 filter_growth_rate: float, 
                 dropout_rate: float, 
                 num_blocks: int, 
                 final_layer: str, 
                 final_channel: int,
                 activation_function):
        """
        init function of CNN model
        
        Args:
            name : str
                some random name for the classifier.  
            
            dropout_rate : float
                to determine the dropout rate.
                
                (designed for the values from 0.1 to 0.5, above 0.5 
                 the model might learn less features)
            
            initial_out_channels : int
                number of output feature maps.
                
                (designed for the values of 16, 32, 64, and 128
                 above 128 the model's complexity increases')
            
            filter_growth_rate : float
                scaling factor that dictates how the number of
                filters or channels increases or decreases as you 
                go deeper into the network.
                
                (designed for the values from 0.5 to 2, above 2
                 the model's complexity increases')
            
            num_blocks : int
                number of layers required to build the network.
            
            final_layer: string
                to determine which final layer to be used
                
                (designed for the layers of linear or nlrl_ao)
            
            final_channel: int
                the input features to the final_layer
                
                (designed for any int values above 0 to 32)
            
            activation_function:
                the activation function that is used in the 
                conv blocks after batchnorm
                
                (eg: ReLU, SiLU, LeakyReLU, etc.)

        Returns
            None.
        """
        super().__init__(name)

        self.logger = get_logger()
        self.logger.info("creating cnn network.")

        self.model = torch.nn.Sequential()
        act = getattr(torch.nn, activation_function)

        for idx in range(num_blocks):
            if idx % 3 == 0:
                out_channels = int(initial_out_channels * filter_growth_rate)
                initial_out_channels *= filter_growth_rate
            self.model.append(ConvBlock(in_channels,
                                        out_channels,
                                        5 if idx == 0 else 3,
                                        0 if idx == 0 else 1,
                                        act))
            if idx % 4 ==0:
                self.model.append(torch.nn.Dropout2d(p=dropout_rate))
            if idx == num_blocks // 2:
                self.model.append(torch.nn.MaxPool2d(2))
            in_channels = out_channels

        self.model.append(ConvBlock(in_channels, 64, 3, 0, act))
        self.model.append(ConvBlock(64, 48, 3, 0, act))
        self.model.append(ConvBlock(48, 32, 3, 0, act))
        self.model.append(torch.nn.AdaptiveMaxPool2d(4))
        self.model.append(torch.nn.Conv2d(32, final_channel, 4))
        self.model.append(torch.nn.Flatten())
        self.model.append(torch.nn.Sigmoid())
        
        if final_layer.lower() == 'linear':
            self.model.append(torch.nn.Linear(final_channel, 10))
        elif final_layer.lower() == 'nlrl':
            self.model.append(NLRL_AO(final_channel, 10))
            self.model.append(InverseSigmoid())
        else:
            raise ValueError(
                f"Invalid value for final_layer: {final_layer}, it should be 'linear', or 'nlrl'")

    def forward(self, ins):
        return self.model(ins)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, act):
        super(ConvBlock, self).__init__()
        self.sequence = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                                            torch.nn.BatchNorm2d(out_channels),
                                            act())
    
    def forward(self, ins):
        return self.sequence(ins)
