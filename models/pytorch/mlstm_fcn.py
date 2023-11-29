'''
Source: Karim, Fazle, et al. "Multivariate LSTM-FCNs for time series classification." Neural networks 116 (2019): 237-245.
(https://arxiv.org/pdf/1801.04503.pdf)

Kuntal's model from G2Net competition
'''

import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=0, dropout=0.2, bias=True):
        super(ConvBlock, self).__init__()
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        # self.bn = nn.BatchNorm1d(out_channels)
        # self.relu = nn.ReLU()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.instance_norm = nn.InstanceNorm1d(out_channels)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.prelu(x)
        x = self.dropout(x)
        return x

class MLSTM_FCN(nn.Module):
    def __init__(self, c_in=3, c_out=1, seq_len=4096,
                 nfs=[128,256,512], kss=[5,11,21], pads=[2,5,10], dropouts=[0.2,0.2,0.2]):
        super(MLSTM_FCN, self).__init__()

        self.conv1 = ConvBlock(c_in, nfs[0], kernel_size=kss[0], padding=pads[0], dropout=dropouts[0])
        self.conv2 = ConvBlock(nfs[0], nfs[1], kernel_size=kss[1], padding=pads[1], dropout=dropouts[1])
        self.conv3 = ConvBlock(nfs[1], nfs[2], kernel_size=kss[2], padding=pads[2], dropout=dropouts[2])
        
        self.attention_data = nn.Conv1d(nfs[2], nfs[1], kernel_size=1)
        self.attention_softmax = nn.Conv1d(nfs[2], nfs[1], kernel_size=1)
        
        self.fc1 = nn.Linear(nfs[1] * seq_len // 4, nfs[2])
        self.instance_norm = nn.InstanceNorm1d(nfs[2])
        self.fc2 = nn.Linear(nfs[2], c_out)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.max_pool1d(x1, kernel_size=2)
        
        x2 = self.conv2(x1)
        x2 = F.max_pool1d(x2, kernel_size=2)
        
        conv3 = self.conv3(x2)
        
        attention_data = self.attention_data(conv3)
        attention_softmax = F.softmax(self.attention_softmax(conv3), dim=2)
        
        multiply_layer = attention_data * attention_softmax
        dense_layer = F.sigmoid(self.instance_norm(self.fc1(multiply_layer.view(multiply_layer.size(0), -1))))
        
        output_layer = self.fc2(dense_layer)
        return output_layer

# class MLSTM_FCN(nn.Module):

#     def __init__(self, c_in=3, c_out=1, seq_len=4096):
#         super(MLSTM_FCN, self).__init__()
        
#         ## Define your architecture here
#         self.conv1 = nn.Conv1d(c_in, 128, kernel_size=5, padding=2)
#         self.instance_norm1 = nn.InstanceNorm1d(128)
#         self.prelu1 = nn.PReLU()
#         self.dropout1 = nn.Dropout(0.2)
        
#         self.conv2 = nn.Conv1d(128, 256, kernel_size=11, padding=5)
#         self.instance_norm2 = nn.InstanceNorm1d(256)
#         self.prelu2 = nn.PReLU()
#         self.dropout2 = nn.Dropout(0.2)
        
#         self.conv3 = nn.Conv1d(256, 512, kernel_size=21, padding=10)
#         self.instance_norm3 = nn.InstanceNorm1d(512)
#         self.prelu3 = nn.PReLU()
#         self.dropout3 = nn.Dropout(0.2)

#         self.attention_data = nn.Conv1d(512, 256, kernel_size=1)
#         self.attention_softmax = nn.Conv1d(512, 256, kernel_size=1)
#         self.fc1 = nn.Linear(256 * input_shape[0] // 4, 512)
#         self.instance_norm4 = nn.InstanceNorm1d(512)
#         self.fc2 = nn.Linear(512, c_out)


        
#     def forward(self, x):
#         x1 = self.prelu1(self.instance_norm1(self.conv1(x)))
#         x1 = self.dropout1(x1)
#         x1 = F.max_pool1d(x1, kernel_size=2)
        
#         x2 = self.prelu2(self.instance_norm2(self.conv2(x1)))
#         x2 = self.dropout2(x2)
#         x2 = F.max_pool1d(x2, kernel_size=2)
        
#         conv3 = self.prelu3(self.instance_norm3(self.conv3(x2)))
#         conv3 = self.dropout3(conv3)
        
#         attention_data = self.attention_data(conv3)
#         attention_softmax = F.softmax(self.attention_softmax(conv3), dim=2)
        
#         multiply_layer = attention_data * attention_softmax
#         dense_layer = F.sigmoid(self.instance_norm4(self.fc1(multiply_layer.view(multiply_layer.size(0), -1))))
        
#         output_layer = self.fc2(dense_layer)
#         return output_layer