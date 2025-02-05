import torch
from ccbdl.network.base import BaseNetwork
from ccbdl.utils.logging import get_logger
from ccbdl.network.nlrl import NLRL_AO, InverseSigmoid


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