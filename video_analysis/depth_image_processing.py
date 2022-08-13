import os

import pyrealsense2 as rs
import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt

from video_analysis.utils import running_mean, butter_bandpass_filter, smooth


class DepthDataAnalyzer:
    def __init__(self, project_path):
        if project_path:
            self.video_path = project_path + "/" + "depth_information.bag"
            self.roi = True
            self.depth_avgs = []
            self.times = []
            self.t0 = None

    def analyze(self):
        try:
            # Create pipeline
            pipeline = rs.pipeline()

            # Create a config object
            config = rs.config()

            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            rs.config.enable_device_from_file(config, self.video_path, repeat_playback=False)

            # Configure the pipeline to stream the depth stream
            # Change this parameters according to the recorded bag file resolution
            config.enable_stream(rs.stream.depth, rs.format.z16, 6)

            # Start streaming from file
            pipeline.start(config)

            profiles = pipeline.get_active_profile()

            dev = profiles.get_device()

            playback = dev.as_playback()
            playback.set_real_time(False)

            # Create opencv window to render image in
            cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

            # Create colorizer object
            colorizer = rs.colorizer()
            c = 0
            # Streaming loop

            avg_depth = []

            while True:
                # Get frameset of depth
                frames = pipeline.wait_for_frames()
                if not frames:
                    break
                # Get depth frame
                depth_frame = frames.get_depth_frame()
                depth_lst = []
                if c > 15:

                    # Colorize depth frame to jet colormap
                    depth_color_raw = np.asanyarray(depth_frame.get_data())
                    depth_color_frame = colorizer.colorize(depth_frame)

                    # Convert depth_frame to numpy array to render image in opencv
                    depth_color_image = np.asanyarray(depth_color_frame.get_data())

                    if self.roi:
                        r = cv2.selectROI(depth_color_image)
                        self.roi = False
                        cv2.destroyAllWindows()

                    selection = depth_color_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                    for x in range(int(r[1]), int(r[1] + r[3])):
                        for y in range(int(r[0]), int(r[0] + r[2])):
                            depth_lst.append(depth_frame.get_distance(y, x))

                    self.depth_avgs.append(np.mean(depth_lst))

                    # print(avg_depth_value)

                    # Render image in opencv window
                    cv2.imshow("Depth Stream", depth_color_image)
                    key = cv2.waitKey(1)
                    # if pressed escape exit program
                    if key == 27:
                        cv2.destroyAllWindows()
                        break
                else:
                    c += 1
        finally:
            return self.depth_avgs
