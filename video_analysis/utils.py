import os

import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import lfilter, butter


def running_mean(x, windowSize):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def generate_plots_from_data(raw_data, project_path, type, export_as_csv=True):
    if not os.path.isdir(project_path + "/analysis"):
        os.mkdir(project_path + "/analysis")

    path_and_fileprefix = project_path + "/analysis/" + type + "-video-processing-"
    detrend = scipy.signal.detrend(raw_data)
    plt.xlabel('frames')
    plt.ylabel('detrended signal amplitude')
    plt.plot(detrend)
    plt.savefig(path_and_fileprefix + "detrend.png")
    plt.clf()

    move = running_mean(detrend, 10)
    plt.xlabel('frames')
    plt.ylabel('moving average')
    plt.plot(move)
    plt.savefig(path_and_fileprefix + "moving-average.png")
    plt.clf()

    y = butter_bandpass_filter(raw_data, 0.08, 0.5, 15, order=3)
    plt.xlabel('frames')
    plt.ylabel('bandpass')
    plt.plot(y)
    plt.savefig(path_and_fileprefix + "bandpass-filter.png")
    plt.clf()

    smoothed = smooth(np.array(raw_data))
    plt.xlabel('frames')
    plt.ylabel('smoothed raw data')
    plt.plot(smoothed)
    plt.savefig(path_and_fileprefix + 'smoothed-raw.png')
    plt.clf()

    plt.xlabel('frames')
    plt.ylabel('raw signal')
    plt.plot(raw_data)
    plt.savefig(path_and_fileprefix + 'raw.png')
    plt.clf()

    if export_as_csv:
        df1 = pd.DataFrame({type + '_raw': raw_data})
        df2 = pd.DataFrame({'smoothed_raw': smoothed.tolist()})
        df3 = pd.DataFrame({'detrend': detrend.tolist()})
        df4 = pd.DataFrame({'moving_average': move.tolist()})
        df5 = pd.DataFrame({'butter_bandpass': y.tolist()})
        df = [df1, df2, df3, df4, df5]
        final = pd.concat(df, ignore_index=True, axis=1)
        final.rename(
            columns={0: type + '_raw', 1: 'smoothed_raw', 2: 'detrend', 3: 'moving_average', 4: 'butter_bandpass'},
            inplace=True)
        final.to_csv(project_path + "/analysis/" + type + '_data.csv')
