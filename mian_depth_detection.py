import cv2
import numpy as np
import pyrealsense2 as rs


class Monitor:

    def __init__(self):
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)

        # pick proper preset
        for i in range(int(preset_range.max)):
            visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
            # print('%02d: %s' % (i, visulpreset))
            if visulpreset == "High Accuracy":
                depth_sensor.set_option(rs.option.visual_preset, i)

        # align depth to color
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        ##################################

        self.start()

    def start(self):

        # init filters
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2)
        spat_filter = rs.spatial_filter()  # Spatial    - edge-preserving spatial smoothing
        temp_filter = rs.temporal_filter()  # Temporal   - reduces temporal noise

        ### get first frame ####

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        # process depth data for more accurate depth detection
        filtered = depth_frame
        filtered = depth_to_disparity.process(filtered)
        filtered = spat_filter.process(filtered)
        filtered = temp_filter.process(filtered)
        filtered = disparity_to_depth.process(filtered)
        depth_frame = filtered.as_depth_frame()
        color_image_aligned = np.asanyarray(aligned_color_frame.get_data())

        # select ROI function
        roi = cv2.selectROI(color_image_aligned)
        x1, y1, x2, y2 = roi
        # destroy it
        cv2.destroyAllWindows()

        ####################
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()
            # process depth data for more accurate depth detection
            filtered = depth_frame
            filtered = depth_to_disparity.process(filtered)
            filtered = spat_filter.process(filtered)
            filtered = temp_filter.process(filtered)
            filtered = disparity_to_depth.process(filtered)
            depth_frame = filtered.as_depth_frame()
            color_image_aligned = np.asanyarray(aligned_color_frame.get_data())

            if not depth_frame:
                print("no depth")
                continue

            cv2.rectangle(color_image_aligned, (x1, y1), (x2 + x1, y2 + y1), (255, 0, 0), 2)


            cv2.namedWindow('RealSense')
            cv2.imshow('RealSense', depth_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

if __name__ == "__main__":
    obj = Monitor()