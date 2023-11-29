import torch
import torch.nn as nn
from ..tsai.tsai.models.ResNet import ResNet
from ..tsai.tsai.models.layers import ConvBlock

# class conv_bn_silu_block(nn.Module):
#     '''
#     This is just my assumption of conv_bn_silu_block needed in the 
#     ConcatBlockConv5 block since the author did not provide a code for it
#     '''
#     def __init__(self, in_ch, out_ch, k, act=nn.SiLU):
#         super().__init__()
#         self.c = nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False)
#         self.bn = nn.BatchNorm1d(out_ch)
#         self.act = act()
#     def forward(self, x):
#         return self.act(self.bn(self.c(x)))

# class ConcatBlockConv5(nn.Module):
#     def __init__(self, in_ch, out_ch, k, act=nn.SiLU):
#         super().__init__()
#         self.c1 = conv_bn_silu_block(in_ch, out_ch, k, act)
#         self.c2 = conv_bn_silu_block(in_ch, out_ch, k * 2, act)
#         self.c3 = conv_bn_silu_block(in_ch, out_ch, k // 2, act)
#         self.c4 = conv_bn_silu_block(in_ch, out_ch, k // 4, act)
#         self.c5 = conv_bn_silu_block(in_ch, out_ch, k * 4, act)
#         self.c6 = conv_bn_silu_block(in_ch * 5 + in_ch, out_ch, 1, act)
#     def forward(self, x):
#         x = torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x), self.c5(x), x], dim=1)
#         x = self.c6(x)
#         return x

class G2NetWinner(nn.Module):
    '''
    Note: this is just my interpretation of the model described here -  https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/discussion/275476
    I am just using a ResNet from tsai and ConvBlock from tsai to encode the data from each detector separately
    instead of using the ConcatBlockConv5 blocks described in the post
    since they do not provide a code for conv_bn_silu_block.
    
    '''
    def __init__(self, c_in, c_out, nf = 64, resnet_kss=[7, 5, 3]):
        super(G2NetWinner, self).__init__()

        ## Define the 1D convolutional layers to encode for data from each detector separately before mixing with a resnet
        
        # self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=nf, kernel_size=3)
        # self.conv2 = nn.Conv1d(in_channels=c_in, out_channels=nf, kernel_size=3)
        # self.conv3 = nn.Conv1d(in_channels=c_in, out_channels=nf, kernel_size=3)

        # self.convblock1 = ConvBlock(1, nf, ks=3)
        # self.convblock2 = ConvBlock(1, nf, ks=3)
        # self.convblock3 = ConvBlock(1, nf, ks=3)

        self.convblocks = nn.ModuleList(
                            [ConvBlock(1, nf, ks=3) \
                                    for i in range(c_in)]
                                )
        
        # self.convblocks = nn.ModuleList(
        #                     [ConcatBlockConv5(1, nf, 3) \
        #                             for i in range(c_in)]
        #                         )

        ## Grab a resnet from tsai
        self.resnet = ResNet(c_in=nf*c_in, c_out=c_out, nf=nf, kss=resnet_kss)

    def forward(self, x):
        ## Split the input into 3 time series
        # ts1, ts2, ts3 = x[:, 0], x[:, 1], x[:, 2]
        
        ## Apply convolutional layers to each time series
        # conv_ts1 = self.conv1(ts1.unsqueeze(1))
        # conv_ts2 = self.conv2(ts2.unsqueeze(1))
        # conv_ts3 = self.conv3(ts3.unsqueeze(1))

        # conv_ts1 = self.convblock1(ts1.unsqueeze(1))
        # conv_ts2 = self.convblock2(ts2.unsqueeze(1))
        # conv_ts3 = self.convblock3(ts3.unsqueeze(1))

        conv_tss = []
        for i in range(x.shape[1]):
            conv_ts = self.convblocks[i](x[:, i].unsqueeze(1))
            conv_tss.append(conv_ts)
            # print(i, conv_tss[i].shape)
        ## Flatten and concatenate the convolutional outputs
        # flat_conv = torch.cat((conv_ts1, conv_ts2, conv_ts3), dim=1)
        flat_conv = torch.cat(conv_tss, dim=1)
        # print(f"flat_conv: {flat_conv.shape}")

        ## Pass the concatenated vector through the ResNet
        output = self.resnet(flat_conv)
        # print(f"output: {output.shape}")
        return output