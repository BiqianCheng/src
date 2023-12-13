import torch.nn.functional as F
from .tsai.tsai.models.TCN import TCN
from .tsai.tsai.models.FCNPlus import FCNPlus
from .tsai.tsai.models.ResNet import ResNet
from .tsai.tsai.models.FCNPlus import FCNPlus
from .tsai.tsai.models.RNNAttention import GRUAttention, LSTMAttention, RNNAttention
from .tsai.tsai.models.MLP import MLP
from .pytorch.tcn import TCN as TCN_og
from .pytorch.g2net_winner import G2NetWinner
from .pytorch.mlstm_fcn import MLSTM_FCN
from pprint import pprint

def get_model(config={}):
    lossfn = F.cross_entropy
    # ##---------------- FCN -------------------------
    # if config.model_name == "FCN":
    #     model = FCN(config.input_channels, 
    #                     config.n_classes)
    #     # model = model_arch
    # ##----------------------------------------------

    ##---------------- FCNPlus -------------------------
    if config['model_name'] == "FCNPlus":
        # print("input channel : ", config['input_channels'])
        # print("sequence is : ", config['seq_len'])
        model = FCNPlus(config['input_channels'],
                        config['n_classes'],
                        layers=config['layers'],
                        kss=config['kernel_sizes'],
                        use_bn=config['batch_norm']
                    )
        # model = model_arch
    ##----------------------------------------------

    ##---------------- Original TCN -------------------------
    elif config['model_name'] == "TCN_og":
        num_hidden_units, levels = config['num_hidden_units'], config['levels']
        channel_sizes = [num_hidden_units] * levels
        kernel_size = config['kernel_size']
        
        ## Original TCN
        model = TCN_og(config['input_channels'], 
                    config['n_classes'], 
                    channel_sizes, 
                    kernel_size=kernel_size, 
                    dropout=config['dropout']
                )
        # model = model_arch
        lossfn = F.nll_loss
    ##----------------------------------------------
    ##---------------- tsai TCN -------------------------
    elif config['model_name'] == "TCN":
        num_hidden_units, levels = config['num_hidden_units'], config['levels']
        layers = [num_hidden_units] * levels
        # kernel_size = config.kernel_size
        # conv_dropout = config.conv_dropout
        # fc_dropout = config.fc_dropout  

        model = TCN(
                    config['input_channels'],
                    config['n_classes'],
                    layers=layers, 
                    ks=config['kernel_size'], 
                    conv_dropout=config['conv_dropout'],
                    fc_dropout=config['fc_dropout']
                )
        # model = model_arch
    ##----------------------------------------------
    ##---------------- ResNet -------------------------
    elif config['model_name'] == "ResNet":
        model = ResNet(config['input_channels'], 
                        config['n_classes'],
                        nf=config['num_filts'],
                        kss=config['kernel_sizes']
                    )
        # model = model_arch
    ##----------------------------------------------
    ##----------------------------------------------
    elif config['model_name'] == "GRUAttention":
        model = GRUAttention(config['input_channels'], 
                            config['n_classes'],
                            config['seq_len'],
                            rnn_layers=config['rnn_layers'],
                            hidden_size=config['hidden_size'],
                            bidirectional=config['bidirectional'],
                            n_heads=config['n_heads'],
                        )
        # model = model_arch
    ##----------------------------------------------
    elif config['model_name'] == "LSTMAttention":
        model = LSTMAttention(config['input_channels'], 
                            config['n_classes'],
                            config['seq_len'],
                            rnn_layers=config['rnn_layers'],
                            hidden_size=config['hidden_size'],
                            bidirectional=config['bidirectional'],
                            n_heads=config['n_heads'],
                        )
        # model = model_arch
    ##----------------------------------------------
    elif config['model_name'] == "RNNAttention":
        model = RNNAttention(config['input_channels'], 
                            config['n_classes'],
                            config['seq_len'],
                            rnn_layers=config['rnn_layers'],
                            hidden_size=config['hidden_size'],
                            bidirectional=config['bidirectional'],
                            n_heads=config['n_heads'],
                        )
        # model = model_arch
    ##----------------------------------------------
    elif config['model_name'] == "G2NetWinner":
        model = G2NetWinner(
                            config['input_channels'],
                            config['n_classes'],
                            nf=config['num_filts'],
                            resnet_kss=config['resnet_kernel_sizes']
                        )
        # model = model_arch
    ## Add more models here
    ##----------------------------------------------
    elif config['model_name'] == "MLP":
        model = MLP(
                        config['input_channels'], 
                        config['n_classes'],
                        config['seq_len'],
                        layers = config['layers'],
                        # ps = config['paddings'],
                        fc_dropout = config['fc_dropout']
                    )
    # ##----------------------------------------------
    # elif config['model_name'] == "MLSTM_FCN":
    #     model = MLSTM_FCN(
    #                         config['input_channels'],
    #                         config['n_classes'],
    #                         nfs=config['num_filts'],
    #                         kss=config['kernel_sizes'],
    #                         pads=config['paddings'],
    #                         dropouts=config['dropouts']
    #                     )
    
    return model, lossfn