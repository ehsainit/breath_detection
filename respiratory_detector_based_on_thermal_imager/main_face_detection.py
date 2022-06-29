import time

import cv2
import imutils
import numpy as np
import scipy
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
from sklearn.decomposition import FastICA


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def draw_rio(frame, roi):
    if roi:
        x, y, w, h = roi

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def minute_passed(oldepoch):
    return time.time() - oldepoch >= 60


if __name__ == '__main__':
    normalized_data = list()
    mean_collection = list()
    timestamps = list()
    result = list()


    # create face detector instance

    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)
    flag = False
    if cap.isOpened():  # try to get the first frame
        val, frame = cap.read()
    else:
        val = False

    new_frame_time = 0
    prev_frame_time = 0
    oldepoch = time.time()
    c = 0
    while val:
        frame = imutils.resize(frame, width=640, height=480)
        if flag is False:
            # select ROI function
            roi = cv2.selectROI(frame)
            x1, y1, x2, y2 = roi
            # destroy it
            cv2.destroyAllWindows()
            flag = True

        for x in range(roi[0], roi[0] + roi[2]):
            for y in range(roi[1], roi[1] + roi[3]):
                normalized_data.append(frame[x][y][0])

        avg = np.asarray(normalized_data)
        np_mean = np.mean(avg)

        mean_collection.append(np_mean)
        timestamps.append(time.time() - oldepoch)
        coll = np.asarray(mean_collection)

        draw_rio(frame, roi)
        c += 1
        if minute_passed(oldepoch):
            fps = float(len(coll)) / (timestamps[-1] - timestamps[0])
            even_times = np.linspace(timestamps[0], timestamps[-1], len(coll))
            # window = np.asarray(coll)
            # https://www.cs.helsinki.fi/u/ahyvarin/whatisica.shtml#:~:text=Independent%20component%20analysis%20(ICA)%20is,a%20large%20database%20of%20samples.
            # window = (window - np.mean(window, axis=0)) / np.std(window, axis=0)  # signal normalization
            # window = (window - np.mean(window, axis=0)) / np.std(window, axis=0)  # signal normalization
            # https://www.frontiersin.org/articles/10.3389/fphys.2018.00948/full
            detrend = scipy.signal.detrend(coll)
            move = running_mean(detrend, 8)

            interpolated = np.interp(even_times, timestamps, detrend)
            interpolated = np.hamming(len(coll)) * interpolated
            norm = interpolated / np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm * 30)
            norm = interpolated / np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm * 30)

            freqs = float(fps) / len(coll) * np.arange(len(coll) / 2 + 1)
            freqs = 60. * freqs

            fft = np.abs(raw) ** 2  # get amplitude spectru
            idx = np.where((freqs > 4.8) & (freqs < 30))  # the range of frequency that BR is supposed to be within
            pruned = fft[idx]
            pfreq = freqs[idx]

            idx2 = np.argmax(pruned)  # max in the range can be HR

            bpm = freqs[idx2]
            result.append(bpm)

            lowcut = 0.08
            highcut = 0.5
            y = butter_bandpass_filter(detrend, lowcut, highcut, fps, order=5)

            print(bpm)



            # plt.show()
        cv2.imshow("preview", frame)

        val, frame = cap.read()

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    cap.release()
