import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.convblock(x))
        else:
            return self.convblock(x)


class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Conditional_Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, c):
        x = self.down(x)
        emb = self.emb_layer(c)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Conditional_Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, c):
        x = self.up(x)
        x = self.conv(x)
        emb = self.emb_layer(c)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Encoder_Decoder(nn.Module):
    def __init__(self, initial_in_channels, final_out_channels, hidden_channels, device="cuda"):
        super().__init__()
        
        self.encoder = Encoder(initial_in_channels, hidden_channels, device="cuda")
        self.decoder = Decoder(final_out_channels, hidden_channels, device="cuda")
        
    def forward(self, ins):
        encoded_images = self.encoder(ins)
        decoded_images = self.decoder(encoded_images)
        return encoded_images, decoded_images


class Encoder(nn.Module):
    def __init__(self, initial_in_channels, hidden_channels, device="cuda"):
        super().__init__()
        self.device = device
        # Downsample
        self.inc = ConvBlock(initial_in_channels, hidden_channels)
        self.encoder1 = Encoder_block(hidden_channels, hidden_channels*2)
        self.encoder2 = Encoder_block(hidden_channels*2, hidden_channels*4)
        self.encoder3 = Encoder_block(hidden_channels*4, hidden_channels*4)

        # bottleneck
        self.bmid1 = ConvBlock(hidden_channels*4, hidden_channels*8)
        self.bmid2 = ConvBlock(hidden_channels*8, hidden_channels*8)
        self.bmid3 = ConvBlock(hidden_channels*8, hidden_channels*4)
    
    def encoding(self, ins):
        #print(f"eins: {ins.shape}")
        x1 = self.inc(ins)
        #print(f"ex1: {x1.shape}")
        x2 = self.encoder1(x1)
        #print(f"ex2: {x2.shape}")
        x3 = self.encoder2(x2)
        #print(f"ex3: {x3.shape}")
        x4 = self.encoder3(x3)
        #print(f"ex4: {x4.shape}")

        x4 = self.bmid1(x4)
        #print(f"ex4: {x4.shape}")
        x4 = self.bmid2(x4)
        #print(f"ex5: {x4.shape}")
        x4 = self.bmid3(x4)
        #print(f"x4: {x4.shape}")
        return x4
    
    def forward(self, ins):
        return self.encoding(ins)


class Decoder(nn.Module):
    def __init__(self, final_out_channels, hidden_channels, device="cuda"):
        super().__init__()
        self.device = device

        # upsample
        self.decoder1 = Decoder_block(hidden_channels*4, hidden_channels*2)
        self.decoder2 = Decoder_block(hidden_channels*2, hidden_channels)
        self.decoder3 = Decoder_block(hidden_channels, hidden_channels)
        self.outc = nn.Conv2d(hidden_channels, final_out_channels, kernel_size=1)
        
    def decoding(self, ins):
        #print(f"dins: {ins.shape}")

        x = self.decoder1(ins)
        #print(f"dx: {x.shape}")
        x = self.decoder2(x)
        #print(f"dx: {x.shape}")
        x = self.decoder3(x)
        #print(f"dx: {x.shape}")
        output = self.outc(x)
        #print(f"out: {output.shape}")
        return output
    
    def forward(self, ins):
        return self.decoding(ins)
    

class Conditional_Encoder_Decoder(nn.Module):
    def __init__(self, initial_in_channels, final_out_channels, hidden_channels, num_labels=10, label_dim=256, device="cuda"):
        super().__init__()
        
        self.label_emb = nn.Embedding(num_labels, label_dim)
        
        self.encoder = Conditional_Encoder(initial_in_channels, hidden_channels, embedded_label=self.label_emb, device="cuda")
        self.decoder = Conditional_Decoder(final_out_channels, hidden_channels, embedded_label=self.label_emb, device="cuda")
        
    def forward(self, ins, num_labels):
        encoded_images = self.encoder(ins, num_labels)
        decoded_images = self.decoder(encoded_images, num_labels)
        return encoded_images, decoded_images


class Conditional_Encoder(nn.Module):
    def __init__(self, initial_in_channels, hidden_channels, embedded_label, device="cuda"):
        super().__init__()
        self.device = device
        # Downsample
        self.inc = ConvBlock(initial_in_channels, hidden_channels)
        self.encoder1 = Conditional_Encoder_block(hidden_channels, hidden_channels*2)
        self.encoder2 = Conditional_Encoder_block(hidden_channels*2, hidden_channels*4)
        self.encoder3 = Conditional_Encoder_block(hidden_channels*4, hidden_channels*4)

        # bottleneck
        self.bmid1 = ConvBlock(hidden_channels*4, hidden_channels*8)
        self.bmid2 = ConvBlock(hidden_channels*8, hidden_channels*8)
        self.bmid3 = ConvBlock(hidden_channels*8, hidden_channels*4)
        
        self.label_emb = embedded_label
    
    def encoding(self, ins, c):
        c = self.label_emb(c)
        # print(f"c: {c.shape}")
        # print(f"ins: {ins.shape}")
        x1 = self.inc(ins)
        # print(f"x1: {x1.shape}")
        x2 = self.encoder1(x1, c)
        # print(f"x2: {x2.shape}")
        x3 = self.encoder2(x2, c)
        # print(f"x3: {x3.shape}")
        x4 = self.encoder3(x3, c)
        # print(f"x4: {x4.shape}")

        x4 = self.bmid1(x4)
        # print(f"x4: {x4.shape}")
        x4 = self.bmid2(x4)
        x4 = self.bmid3(x4)
        return x4
    
    def forward(self, ins, c):
        return self.encoding(ins, c)


class Conditional_Decoder(nn.Module):
    def __init__(self, final_out_channels, hidden_channels, embedded_label, device="cuda"):
        super().__init__()
        self.device = device
        # upsample
        self.decoder1 = Conditional_Decoder_block(hidden_channels*4, hidden_channels*2)
        self.decoder2 = Conditional_Decoder_block(hidden_channels*2, hidden_channels)
        self.decoder3 = Conditional_Decoder_block(hidden_channels, hidden_channels)
        self.outc = nn.Conv2d(hidden_channels, final_out_channels, kernel_size=1)
        
        self.label_emb = embedded_label
        
    def decoding(self, ins, c):
        # print(f"ins: {ins.shape}")
        
        c = self.label_emb(c)
        # print(f"c: {c.shape}")

        x4 = ins
        # print(f"x4: {x4.shape}")

        x = self.decoder1(x4, c)
        # print(f"x: {x.shape}")
        x = self.decoder2(x, c)
        # print(f"x: {x.shape}")
        x = self.decoder3(x, c)
        # print(f"x: {x.shape}")
        output = self.outc(x)
        # print(f"out: {output.shape}")
        return output
    
    def forward(self, ins, c):
        return self.decoding(ins, c)




# import torch
# from ccbdl.network.base import BaseNetwork
# from ccbdl.utils.logging import get_logger


# class ConvBlock(torch.nn.Module):
#     def __init__(self, shape, in_c, out_c, activation=None):
#         super(ConvBlock, self).__init__()
#         self.sequence = torch.nn.Sequential(torch.nn.LayerNorm(shape),
#                                             torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
#                                             torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
#                                             torch.nn.ReLU() if activation is None else activation())
    
#     def forward(self, ins):
#         return self.sequence(ins)


# class Encoder_Decoder(BaseNetwork):
#     def __init__(self, name, in_channels, hidden_channels, 
#                  activation_function, num_layers, block_layers):
#         super().__init__(name)
#         self.encoder = Encoder(in_channels, hidden_channels, activation_function, num_layers, block_layers)
        
#         in_channels_de = (hidden_channels//2)*(2**num_layers)
#         out_channels_de = (hidden_channels//2)*(2**num_layers)
        
#         self.decoder = Decoder(in_channels_de, out_channels_de, activation_function, num_layers, block_layers)

#     def forward(self, ins):
#         encoded_images = self.encoder(ins)
#         decoded_images = self.decoder(encoded_images)
#         return encoded_images, decoded_images


# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, 
#                  activation_function, num_layers, block_layers):
#         super(Encoder, self).__init__()

#         self.logger = get_logger()
#         self.logger.info("Encoder network.")
        
#         act = getattr(torch.nn, activation_function)

#         # Initialize lists to store down-sampling blocks
#         self.encoder_blocks = torch.nn.Sequential()
        
#         self.downs = torch.nn.Sequential()

#         # Add down-sampling blocks and corresponding layers
#         for i in range(num_layers):
#             if i == 0:
#                 out_channels = hidden_channels
#             else:
#                 out_channels = out_channels*2
                
#             encoder_block = []
#             for j in range(block_layers):
#                 if j == 0:
#                     encoder_block.append(ConvBlock((in_channels, 32 // (2 ** i), 32 // (2 ** i)), in_channels, out_channels, act))
#                 else:
#                     encoder_block.append(ConvBlock((out_channels, 32 // (2 ** i), 32 // (2 ** i)), out_channels, out_channels, act))
#             encoder_block = torch.nn.Sequential(*encoder_block)
            
#             down = torch.nn.Sequential(torch.nn.Conv2d(out_channels, out_channels, 4, 2, 1),
#                                        act())
                                       
#             self.encoder_blocks.append(encoder_block)
#             self.downs.append(down)
#             in_channels = out_channels

#         # Add middle block and corresponding layer
#         b_mid = []
#         for _ in range(block_layers):
#             b_mid.append(ConvBlock((out_channels, 32 // (2 ** num_layers), 32 // (2 ** num_layers)), out_channels, out_channels, act))
#         self.b_mid = torch.nn.Sequential(*b_mid)
    
#     def encoding(self, ins):
#         # Down-sampling path
#         out = ins
#         for i, encoder_block in enumerate(self.encoder_blocks):
#             if i == 0:
#                 out = encoder_block(out)
#             else:
#                 out = encoder_block(self.downs[i-1](out))

#         # Middle block
#         out = self.b_mid(self.downs[i](out))
#         return out
    
#     def forward(self, images):
#         return self.encoding(images)


# class Decoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, 
#                  activation_function, num_layers, block_layers):
#         super(Decoder, self).__init__()

#         self.logger = get_logger()
#         self.logger.info("Decoder network.")
        
#         act = getattr(torch.nn, activation_function)

#         # Initialize lists to store up-sampling blocks
#         self.decoder_blocks = torch.nn.Sequential()
        
#         self.ups = torch.nn.Sequential()
        
#         # Add up-sampling blocks and corresponding layers
#         for i in reversed(range(num_layers)):
#             if i == (num_layers - 1):
#                 in_channels = out_channels  
#             up = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
#                                      act())
                        
#             if i != 0:
#                 decoder_block = []
#                 for j in range(block_layers):
#                     if j == 0:
#                         decoder_block.append(ConvBlock((in_channels, 32 // (2 ** i), 32 // (2 ** i)), in_channels, out_channels, act))
#                     elif j == 1:
#                         decoder_block.append(ConvBlock((out_channels, 32 // (2 ** i), 32 // (2 ** i)), out_channels, out_channels//2, act))
#                     else:
#                         decoder_block.append(ConvBlock((out_channels//2, 32 // (2 ** i), 32 // (2 ** i)), out_channels//2, out_channels//2, act))
#                 decoder_block = torch.nn.Sequential(*decoder_block)
                
#                 in_channels = out_channels//2
#                 out_channels = out_channels//2
#             else:
#                 decoder_block = []
#                 for j in range(block_layers):
#                     if j == 0:
#                         decoder_block.append(ConvBlock((in_channels, 32 // (2 ** i), 32 // (2 ** i)), in_channels, out_channels, act))
#                     else:
#                         decoder_block.append(ConvBlock((out_channels, 32 // (2 ** i), 32 // (2 ** i)), out_channels, out_channels, act))
#                 decoder_block = torch.nn.Sequential(*decoder_block)
                
#             self.decoder_blocks.append(decoder_block)
#             self.ups.append(up)

#         # Add output convolution
#         self.conv_out = torch.nn.Conv2d(out_channels, 1, 3, 1, 1)
    
#     def decoding(self, ins):
#         out = ins
#         # Up-sampling path
#         for i, (decoder_block) in enumerate(self.decoder_blocks):
#             out = decoder_block(self.ups[i](out))

#         # Output convolution
#         out = self.conv_out(out)
#         return out
    
#     def forward(self, images):
#         return self.decoding(images)


# class Conditional_Encoder_Decoder(BaseNetwork):
#     def __init__(self, name, in_channels, hidden_channels, 
#                  activation_function, num_layers, block_layers, label_emb_dim, num_classes):
#         super().__init__(name)
#         self.encoder = Conditional_Encoder(in_channels, hidden_channels, activation_function, num_layers, block_layers,
#                                num_classes, label_emb_dim)
        
#         in_channels_de = (hidden_channels//2)*(2**num_layers)
#         out_channels_de = (hidden_channels//2)*(2**num_layers)
        
#         self.decoder = Conditional_Decoder(in_channels_de, out_channels_de, activation_function, num_layers, block_layers,
#                                num_classes, label_emb_dim)

#     def forward(self, ins, labels):
#         encoded_images = self.encoder(ins, labels)
#         decoded_images = self.decoder(encoded_images, labels)
#         return encoded_images, decoded_images


# class Conditional_Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, 
#                  activation_function, num_layers, block_layers, num_classes, label_emb_dim):
#         super(Conditional_Encoder, self).__init__()

#         self.logger = get_logger()
#         self.logger.info("Conditional Encoder network.")
        
#         # Label embedding
#         self.label_embed = torch.nn.Embedding(num_classes, label_emb_dim)
        
#         act = getattr(torch.nn, activation_function)

#         # Initialize lists to store down-sampling blocks
#         self.encoder_blocks = torch.nn.Sequential()
        
#         self.downs = torch.nn.Sequential()
        
#         # Initialize list to store label embedding layers
#         self.lb_down_list = torch.nn.Sequential()

#         # Add down-sampling blocks and corresponding label embedding layers
#         for i in range(num_layers):
#             if i == 0:
#                 out_channels = hidden_channels
#                 lb_layer = self._make_lb(label_emb_dim, 1)
#             else:
#                 lb_layer = self._make_lb(label_emb_dim, in_channels)
#                 out_channels = out_channels*2
                
#             encoder_block = []
#             for j in range(block_layers):
#                 if j == 0:
#                     encoder_block.append(ConvBlock((in_channels, 32 // (2 ** i), 32 // (2 ** i)), in_channels, out_channels, act))
#                 else:
#                     encoder_block.append(ConvBlock((out_channels, 32 // (2 ** i), 32 // (2 ** i)), out_channels, out_channels, act))
#             encoder_block = torch.nn.Sequential(*encoder_block)
            
#             down = torch.nn.Sequential(torch.nn.Conv2d(out_channels, out_channels, 4, 2, 1),
#                                        act())
                                       
#             self.encoder_blocks.append(encoder_block)
#             self.downs.append(down)
#             in_channels = out_channels
            
#             self.lb_down_list.append(lb_layer)

#         # Add middle block and corresponding time embedding layer
#         self.lb_mid = self._make_lb(label_emb_dim, in_channels)
#         b_mid = []
#         for _ in range(block_layers):
#             b_mid.append(ConvBlock((out_channels, 32 // (2 ** num_layers), 32 // (2 ** num_layers)), out_channels, out_channels, act))
#         self.b_mid = torch.nn.Sequential(*b_mid)
    
#     def _make_lb(self, dim_in, dim_out):
#         return torch.nn.Sequential(
#             torch.nn.Linear(dim_in, dim_out),
#             torch.nn.SiLU(),
#             torch.nn.Linear(dim_out, dim_out)
#         )

#     def encoding(self, ins, lbs):
#         lbs = self.label_embed(lbs)
#         n = len(ins)

#         # Down-sampling path
#         out = ins
#         for i, (encoder_block, lb_layer) in enumerate(zip(self.encoder_blocks, self.lb_down_list)):
#             if i == 0:
#                 out = encoder_block(out + lb_layer(lbs).reshape(n, -1, 1, 1))
#             else:
#                 out = encoder_block(self.downs[i-1](out) + lb_layer(lbs).reshape(n, -1, 1, 1))

#         # Middle block
#         out = self.b_mid(self.downs[i](out) + self.lb_mid(lbs).reshape(n, -1, 1, 1))
#         return out
    
#     def forward(self, images, labels):
#         return self.encoding(images, labels)


# class Conditional_Decoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, 
#                  activation_function, num_layers, block_layers, num_classes, label_emb_dim):
#         super(Conditional_Decoder, self).__init__()

#         self.logger = get_logger()
#         self.logger.info("Conditional_Decoder network.")
        
#         act = getattr(torch.nn, activation_function)
        
#         # Label embedding
#         self.label_embed = torch.nn.Embedding(num_classes, label_emb_dim)

#         # Initialize lists to store up-sampling blocks
#         self.decoder_blocks = torch.nn.Sequential()
#         self.ups = torch.nn.Sequential()
        
#         # Initialize list to store label embedding layers
#         self.lb_up_list = torch.nn.Sequential()
        
#         # Add up-sampling blocks and corresponding time embedding layers
#         for i in reversed(range(num_layers)):
#             if i == (num_layers - 1):
#                 in_channels = out_channels
                
#             lb_layer = self._make_lb(label_emb_dim, in_channels)
#             up = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
#                                      act())
                        
#             if i != 0:
#                 decoder_block = []
#                 for j in range(block_layers):
#                     if j == 0:
#                         decoder_block.append(ConvBlock((in_channels, 32 // (2 ** i), 32 // (2 ** i)), in_channels, out_channels, act))
#                     elif j == 1:
#                         decoder_block.append(ConvBlock((out_channels, 32 // (2 ** i), 32 // (2 ** i)), out_channels, out_channels//2, act))
#                     else:
#                         decoder_block.append(ConvBlock((out_channels//2, 32 // (2 ** i), 32 // (2 ** i)), out_channels//2, out_channels//2, act))
#                 decoder_block = torch.nn.Sequential(*decoder_block)
                
#                 in_channels = out_channels//2
#                 out_channels = out_channels//2
#             else:
#                 decoder_block = []
#                 for j in range(block_layers):
#                     if j == 0:
#                         decoder_block.append(ConvBlock((in_channels, 32 // (2 ** i), 32 // (2 ** i)), in_channels, out_channels, act))
#                     else:
#                         decoder_block.append(ConvBlock((out_channels, 32 // (2 ** i), 32 // (2 ** i)), out_channels, out_channels, act))
#                 decoder_block = torch.nn.Sequential(*decoder_block)
                
#             self.decoder_blocks.append(decoder_block)
#             self.ups.append(up)

#             self.lb_up_list.append(lb_layer)

#         # Add output convolution
#         self.conv_out = torch.nn.Conv2d(out_channels, 1, 3, 1, 1)
    
#     def _make_lb(self, dim_in, dim_out):
#         return torch.nn.Sequential(
#             torch.nn.Linear(dim_in, dim_out),
#             torch.nn.SiLU(),
#             torch.nn.Linear(dim_out, dim_out)
#         )

#     def decoding(self, ins, lbs):
#         lbs = self.label_embed(lbs)
#         n = len(ins)
#         out = ins

#         # Up-sampling path
#         for i, (decoder_block, lb_layer) in enumerate(zip(self.decoder_blocks, self.lb_up_list)):
#             out = decoder_block(self.ups[i](out) + lb_layer(lbs).reshape(n, -1, 1, 1))

#         # Output convolution
#         out = self.conv_out(out)
#         return out
    
#     def forward(self, images, labels):
#         return self.decoding(images, labels)
