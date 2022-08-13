import time

import cv2
import scipy

from scipy import signal

import matplotlib.pyplot as plt

import skimage.io
import skimage.color
import skimage.filters
import numpy as np


class ThermalDataAnalyzer:
    def __init__(self, project_path):
        if project_path:
            self.video_path = project_path + "/" + "V00P00A00C00_thermo.avi"
            self.blurred_image = None
            self.roi = True
            self.temps = []
            self.times = []
            self.t0 = None

    def analyze(self):
        cap = cv2.VideoCapture(self.video_path)
        self.t0 = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if self.roi:
                    r = cv2.selectROI(frame)
                    self.roi = False
                    cv2.destroyAllWindows()

                # Thresholding-noise filter on gray scale according to experimental mapping between
                # thermal data and gray scale values
                blurred_image = skimage.filters.gaussian(frame, sigma=1.0)
                if self.blurred_image is None:
                    self.blurred_image = blurred_image
                # create a histogram of the blurred grayscale image
                # create a mask based on the threshold
                t = 0.6
                blurred_image = blurred_image > t  # The frame is ready and already captured
                selection = frame.copy()
                selection[~blurred_image] = 0
                cv2.imshow("previewName", selection)

                selection = selection[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                # try with avarge
                avg_grayscale_value = np.mean(selection)
                # print(avg_grayscale_value)
                factor = 0.08038
                # convert to temperature
                avg_temp = (260 - avg_grayscale_value) * factor + 27.00
                self.temps.append(avg_temp)

                self.times.append(time.time() - self.t0)

                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                break
                # check the temperature and correct equations or maybe the threshold try to have a better signal
                # name = "thermal-video-processing-"
                # detrend = scipy.signal.detrend(self.temps)
                # plt.plot(detrend)
                # plt.savefig(name + "detrend.png")
                # plt.clf()
                #
                # move = running_mean(detrend, 10)
                # plt.plot(move)
                # plt.savefig(name + "move.png")
                # plt.clf()
                #
                # y = butter_bandpass_filter(move, 0.08, 0.5, 15, order=3)
                # plt.plot(y)
                # plt.savefig(name + "bandpass-filter.png")
                # plt.clf()
                #
                # y = smooth(np.array(self.temps))
                # plt.plot(y)
                # plt.savefig(name + 'smoothed-bandpass.png')
                # # plt.show()

        cv2.destroyWindow("previewName")
        return self.temps

    def plot_blurred_histrogram(self):
        if self.blurred_image is None:
            print("[!] run analyze first")

        histogram, bin_edges = np.histogram(self.blurred_image, bins=256, range=(0.0, 1.0))

        fig, ax = plt.subplots()
        plt.plot(bin_edges[0:-1], histogram)
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixels")
        plt.xlim(0, 1.0)
        plt.show()
