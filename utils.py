'''
File Created: Wednesday, 24th May 2023 5:32:56 pm
Author: Rutuja Gurav (rgura001@ucr.edu)
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch

# from constantQ.timeseries import TimeSeries as cQts

# import gwpy
# print("GwPy version: {}".format(gwpy.__version__))
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.time import from_gps, to_gps
from gwpy.signal import filter_design
from gwpy.plot import Plot

from sklearn.model_selection import train_test_split

def convert_image_id_2_path(image_id: str, 
                            is_train: bool = True, 
                            data_dir="/data/rgura001/ML4GWsearch/g2net-gravitational-wave-detection", ) -> str:
    '''
    Source - https://www.kaggle.com/code/ihelon/g2net-eda-and-modeling
    Convenience function to fetch the filepath based on the sample ID.
    '''
    # print("# {}".format(image_id))
    folder = "train" if is_train else "test"
    return data_dir+"/{}/{}/{}/{}/{}.npy".format(
        folder, image_id[0], image_id[1], image_id[2], image_id 
    )

def train_val_test_split(ids_df, train_ratio = 0.75,
                        validation_ratio = 0.15,
                        test_ratio = 0.10,
                        rng_seed=42):
    
    ids_train_df, ids_test_df = train_test_split(ids_df, 
                                                    test_size=1 - train_ratio, 
                                                    shuffle=True, 
                                                    random_state=rng_seed)


    ids_val_df, ids_test_df = train_test_split(ids_test_df,
                                                test_size=test_ratio/(test_ratio + validation_ratio), 
                                                shuffle=True, 
                                                random_state=rng_seed)
    
    return ids_train_df, ids_val_df, ids_test_df

def visualize_sample(
    _id, 
    target, 
    colors=("black", "red", "green"), 
    signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo")
):
    '''
    Source - https://www.kaggle.com/code/ihelon/g2net-eda-and-modeling
    Generic convenience function to visualize a sample.
    '''
    path = convert_image_id_2_path(_id)
    x = np.load(path)
    plt.figure(figsize=(16, 7))
    for i in range(3):
        plt.subplot(4, 1, i + 1)
        plt.plot(x[i], color=colors[i])
        plt.legend([signal_names[i]], fontsize=12, loc="lower right")
        
        plt.subplot(4, 1, 4)
        plt.plot(x[i], color=colors[i])
    
    plt.subplot(4, 1, 4)
    plt.legend(signal_names, fontsize=12, loc="lower right")

    plt.suptitle(f"id: {_id} target: {target}", fontsize=16)
    plt.show()

    ## Q-transform
    # fig, ax = plt.subplots(1,3, figsize=(30,7))
    # for i in range(3):
    #     ts = cQts(x[i,:], dt = 1/2048.0, unit='s', name=signal_names[i])
    #     qt = ts.q_transform(search=None)
    #     ax[i].imshow(qt)
    #     ax[i].set_xscale('seconds')
    #     ax[i].set_yscale('log')
    #     ax[i].set_xlabel("Time [s]", fontsize=20)
    #     ax[i].grid(True, axis='y', which='both')
    #     ax[i].set_title(signal_names[i], fontsize=20)
    #     if i==0:
    #         ax[i].set_ylabel('Frequency [Hz]', fontsize=20)
    #     ax[i].colorbar(cmap='viridis').set_label(label='Normalized energy', size=20)
    # plt.show()

# def preprocess_data(x, z_norm=False, highpass=False, order=5, fc=20.48):
#     if highpass: ## From winning entry in the g2net kaggle competition: order=5, fc=20.48
#         # print(f"Highpass filtering at {fc} Hz...")
#         b, a = scipy.signal.butter(order, fc, 
#                                 btype='highpass', 
#                                 fs=2048.0)
#         x = scipy.signal.filtfilt(b, a, x)
#     if z_norm:
#         ## z-norm values 
#         x = (x - np.mean(x)) / np.std(x)
    
#     return x

# def convert_to_tsdict(_id=None, 
#                       x = None,
#                       target=None,
#                       z_norm=False,
#                       highpass=False, 
#                       whiten=False, fdur = 0.25,
#                       signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo"),
#                       sample_rate = 2048.0
#                     ):
#     '''
#     Converts data from numpy array into gwpy.TimeSeriesDict for easy visualization.
#     '''
#     if x == None:
#         # print(f"Loading data from file for id={_id}, target={target}...")
#         path = convert_image_id_2_path(_id)
#         x = np.load(path)
    
#     # print(x.shape)
#     # print("DISCLAIMER: The timestamp used here (2000-01-01) is a placeholder!")
#     t0=to_gps('2000-01-01T00:00:00')
    
#     ts_h=TimeSeries(data=preprocess_data(x[0,:], 
#                                          z_norm=z_norm, 
#                                          highpass=highpass,
#                                         ), 
#                     t0=t0, 
#                     sample_rate=sample_rate, 
#                     name="{} (id {}: target {} )".format(signal_names[0], _id, target))
    
#     ts_l=TimeSeries(data=preprocess_data(x[1,:],
#                                          z_norm=z_norm, 
#                                          highpass=highpass,
#                                         ), 
#                     t0=t0, 
#                     sample_rate=sample_rate, 
#                     name="{} (id {}: target {} )".format(signal_names[1], _id, target))
    
#     ts_v=TimeSeries(data=preprocess_data(x[2,:],
#                                          z_norm=z_norm, 
#                                          highpass=highpass,
#                                         ), 
#                     t0=t0, 
#                     sample_rate=sample_rate, 
#                     name="{} (id: {} target: {})".format(signal_names[2], _id, target))
#     # if highpass:
#     #     print("Highpass filtering at 20 Hz...")
#     #     ts_h = ts_h.highpass(20)
#     #     ts_l = ts_l.highpass(20)
#     #     ts_v = ts_v.highpass(20)
#     if whiten:
#         # print("Whitening...")
#         ts_h = ts_h.whiten(fduration=fdur)
#         ts_l = ts_l.whiten(fduration=fdur)
#         ts_v = ts_v.whiten(fduration=fdur)
#         # print(from_gps(ts_h.span[0]), from_gps(ts_h.span[1]))
#         # print("Cropping...")
#         ts_h = ts_h.crop(start=ts_h.span[0]+(0.5*fdur), end=ts_h.span[1]-(0.5*fdur))
#         ts_l = ts_l.crop(start=ts_l.span[0]+(0.5*fdur), end=ts_l.span[1]-(0.5*fdur))
#         ts_v = ts_v.crop(start=ts_v.span[0]+(0.5*fdur), end=ts_v.span[1]-(0.5*fdur))
#         # print(from_gps(ts_h.span[0]), from_gps(ts_h.span[1]))

#     return TimeSeriesDict({
#         signal_names[0]: ts_h,
#         signal_names[1]: ts_l,
#         signal_names[2]: ts_v
#     })

def z_norm_ts(x):
    x = (x - np.mean(x)) / np.std(x)
    return x

def scale_ts(x): # From Kuntal
    # Scaling the data in the [-1,1] range so as to transform to polar co-ordinates
    # for Gramian angular fields.
    min_stamp = np.amin(x)
    max_stamp = np.amax(x)
    x = (2*x - max_stamp - min_stamp)/(max_stamp - min_stamp)

    # Checking for any floating point accuracy.
    x = np.where(x >= 1., 1., x)
    x = np.where(x <= -1., -1., x)

    return x

def whiten_ts(x): # From Kuntal
    # Assuming x has shape (length, channels)
    hann = torch.hann_window(x.shape[0], periodic=True, dtype=torch.float32)
    whitened_channels = []
    
    for ch in range(x.shape[1]):
        single_channel_data = torch.tensor(x[:, ch], dtype=torch.float32)
        spec = torch.fft.fft(single_channel_data * hann)
        mag = torch.sqrt(torch.real(spec * torch.conj(spec)))
        whitened_channel = torch.fft.ifft(spec / mag) * torch.sqrt(torch.tensor(len(single_channel_data) / 2, dtype=torch.float32))
        whitened_channels.append(whitened_channel.real)
        
    return torch.stack(whitened_channels, dim=1)

def apply_highpass(x, order=5, fc=20.48):
    '''
    x: A numpy array with shape (num_channels, sequence_length)
    '''
    b, a = scipy.signal.butter(order, fc, 
                                btype='highpass', 
                                fs=2048.0)
    
    if len(x.shape) == 1:
        # x = scipy.signal.tukey(x.shape[0], 0.2)
        x = scipy.signal.filtfilt(b, a, x)
    else:
        num_channels = x.shape[0]
        for ch in range(num_channels):
            # x[ch,:] = scipy.signal.tukey(x[ch,:].shape[0], 0.2)
            x[ch,:] = scipy.signal.filtfilt(b, a, x[ch,:])
    return x

def apply_bandpass(x, lf=20, hf=500, order=8, sr=2048): # From Kuntal
    # Generate the filter coefficients
    sos = scipy.signal.butter(order, [lf, hf], btype='bandpass', output='sos', fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))

    # Apply the filter
    if len(x.shape) == 1:
        # Single channel
        x *= scipy.signal.tukey(x.shape[0], 0.2)
        x = scipy.signal.sosfiltfilt(sos, x) / normalization
    else:
        # Multi-channel
        num_channels = x.shape[0]
        for ch in range(num_channels):
            x[ch, :] *= scipy.signal.tukey(x[ch, :].shape[0], 0.2)
            x[ch, :] = scipy.signal.sosfiltfilt(sos, x[ch, :]) / normalization

    return x

def get_sample_as_tsdict(_id=None, 
                      x = None,
                      target=None,
                      signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo"),
                      sample_rate = 2048.0):
    '''
    Reads data into gwpy.TimeSeriesDict.
    '''
    if x == None:
        # print(f"Loading data from file for id={_id}, target={target}...")
        path = convert_image_id_2_path(_id)
        x = np.load(path)
    
    # print(x.shape)
    # print("DISCLAIMER: The timestamp used here (2000-01-01) is a placeholder!")
    t0=to_gps('2000-01-01T00:00:00')
    
    ts_h=TimeSeries(data=x[0,:], 
                    t0=t0, 
                    sample_rate=sample_rate, 
                    name="{} (id {}: target {} )".format(signal_names[0], _id, target))
    
    ts_l=TimeSeries(data=x[1,:],              
                    t0=t0, 
                    sample_rate=sample_rate, 
                    name="{} (id {}: target {} )".format(signal_names[1], _id, target))
    
    ts_v=TimeSeries(data=x[2,:],                 
                    t0=t0, 
                    sample_rate=sample_rate, 
                    name="{} (id: {} target: {})".format(signal_names[2], _id, target))

    return TimeSeriesDict({
        signal_names[0]: ts_h,
        signal_names[1]: ts_l,
        signal_names[2]: ts_v
    })

def preprocess_tsdict(ts_dict, scale=True, bandpass=True, z_norm=False, highpass=False, whiten=False):
    ## Kuntal's preprocessing steps
    ## scale=True, bandpass=True, highpass=False, whiten=False, z_norm=False

    ## Rutuja's preprocessing steps
    ## scale=False, bandpass=False, highpass=True, whiten=False, z_norm=True
    if bandpass and highpass:
        raise ValueError("Cannot apply both bandpass and highpass filters!")
    scale=True
    bandpass=True
    z_norm=False
    highpass=False
    for ts_name, ts in ts_dict.items():
        ts_val = ts.value
        if z_norm and highpass:
            ts_val = z_norm_ts(ts_val)
            ts_val = apply_highpass(ts_val)
        elif scale and bandpass:
            ts_val = scale_ts(ts_val)
            ts_val = apply_bandpass(ts_val)
        elif bandpass:
            ts_val = apply_bandpass(ts_val)
        elif highpass:
            ts_val = apply_highpass(ts_val)
        else:
            print("No preprocessing applied!")
        
        # ## normalize values in ts
        # if z_norm: ts_val = z_norm_ts(ts_val)
        # if scale: ts_val = scale_ts(ts_val)

        # ## freq filtering of ts
        # if highpass: ts_val = apply_highpass(ts_val)
        # if bandpass: ts_val = apply_bandpass(ts_val)
        
        # # if whiten: ts_val = whiten_ts(ts_val)

        ts_dict[ts_name] = TimeSeries(data=ts_val, 
                                      t0=ts.t0, sample_rate=ts.sample_rate, 
                                      name=ts_name)
    return ts_dict

def get_qtransform(ts, qrange=(8,32), frange=(20,400), logf=True, whiten=False):
    return ts.q_transform(qrange=qrange, 
                    frange=frange, 
                    logf=logf, 
                    whiten=whiten)
    
def plot_sample_ts(_id=None,
                   x = None,
                   target=None,
                   z_norm=False,
                   highpass=False,
                   bandpass=False,
                   scale=False,
                   whiten=False,
                   ts_whiten=False,
                   qt_whiten=False, qrange=(8,32), frange=(20,400),
                   signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo"),
                   sample_rate = 2048.0
                ):
    '''
    Plots the gwpy.TimeSeriesDict of the sample.
    '''
    # ts_dict = convert_to_tsdict(_id=_id, x=x, target=target, 
    #                             z_norm=z_norm, highpass=highpass,
    #                             whiten=ts_whiten,
    #                             signal_names=signal_names)

    ts_dict = get_sample_as_tsdict(_id=_id, x=x, target=target)
    if highpass or z_norm or bandpass or scale:
        ts_dict = preprocess_tsdict(ts_dict, 
                                    scale=scale, 
                                    bandpass=bandpass, 
                                    highpass=highpass, 
                                    z_norm=z_norm)

    # plot = ts_dict.plot(label='key', title="id {}: target {}".format(_id, target))
    # plot.legend()
    # plot.show()

    asd_dict = {}
    psd_dict = {}
    for i, (ts_ifo, ts) in enumerate(ts_dict.items()):
        # start, end = ts.span
        plot = ts.plot()
        axs = plot.gca()
        axs.set_ylabel('Amplitude [strain]')
        axs.set_title("{} (id {}: target {})".format(ts_ifo, _id, target))
        plot.show()

        # plot = ts.q_transform(
        #                     qrange=(8,32), 
        #                     frange=(20,400), 
        #                     logf=True, 
        #                     whiten=qt_whiten).plot()

        plot = get_qtransform(ts, 
                                    frange=frange,
                                    ).plot()
            
        axs = plot.gca()
        axs.set_title("{} (id {}: target {})".format(ts_ifo, _id, target))
        axs.set_xscale('seconds')
        axs.set_yscale('log')
        axs.set_ylim(20,400)
        axs.set_ylabel('Frequency [Hz]')
        axs.grid(True, axis='y', which='both')
        axs.colorbar(cmap='viridis', label='Normalized energy')
        plot.show()

        # plot = ts.asd(0.5, 0).plot()
        # ax = plot.gca()
        # ax.set_xlim(40, 0.5*sample_rate)
        # # ax.set_ylim(1e-23, 3e-20)
        # ax.set_ylabel(r'GW strain ASD [strain$/\sqrt{\mathrm{Hz}}$]')
        # plot.show()

        asd_dict[ts_ifo] = ts.asd(0.5, 0.01)
        psd_dict[ts_ifo] = ts.psd(0.5, 0.01)
    
    asd_dict = TimeSeriesDict(asd_dict)
    plot = asd_dict.plot(label='key')
    ax = plot.gca()
    ax.set_xlim(20, 0.5*sample_rate)
    # ax.set_ylim(1e-25, 1e-22)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'GW strain ASD [strain$/\sqrt{\mathrm{Hz}}$]')
    plot.legend()
    plot.show()

    return asd_dict

    # psd_dict = TimeSeriesDict(psd_dict)
    # plot = psd_dict.plot(label='key')
    # ax = plot.gca()
    # ax.set_xlim(20, 0.5*sample_rate)
    # # ax.set_ylim()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylabel(r'GW strain PSD [strain$/\sqrt{\mathrm{Hz}}$]')
    # plot.legend()
    # plot.show()
    
def plot_training_metrics(history=None, RUN_DIR=None, save=True):
    """
    Plots and saves training metrics vs epochs plots in your RUN_DIR.
    Args: 
        history: A dictionary with keys as metric names. 
                Train metric names are just metric names. 
                Val metric names are appended with prefix 'val_'
        RUN_DIR: Location to save the plots.
    """
    metrics = [metric.split('_',1)[-1] for metric in history.keys()]

    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(range(len(history['train_'+metric])), history['train_'+metric])
        if metric != 'lr':
            ax.plot(range(len(history['val_'+metric])), history['val_'+metric])
        # if 'loss' in metric:
        #     plt.ylim(0.0, 1.0)
        ax.set_ylabel(metric)
        ax.set_xlabel('epoch')
        if metric != 'lr':
            plt.legend(['train', 'validation'], loc='best')
        if save:
            plt.savefig(RUN_DIR+f'/{metric}.png')
            plt.close()
        else:
            plt.show()
        

        
    



