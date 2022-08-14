import time
import os
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import uuid

unique_dirname = str(uuid.uuid4())
color_path = unique_dirname + "/" + 'V00P00A00C00_rgb.avi'
depth_path = unique_dirname + "/" + 'V00P00A00C00_depth.avi'
thermo_path = unique_dirname + "/" + 'V00P00A00C00_thermo.avi'
thermo_raw_path = unique_dirname + "/" + 'V00P00A00C00_thermo_raw.avi'


start_time = 0


class camThread(threading.Thread):
    def __init__(self, previewName, camID, kind):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.kind = kind

    def run(self):
        print("Starting " + self.previewName)
        if self.kind == "thermal":
            cam_preview_thermal(self.previewName, self.camID)

        if self.kind == "depth":
            cam_preview_depth(self.previewName)


def cam_preview_depth(previewName):
    global colorwriter
    global depthwriter
    global unique_dirname
    global start_time
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    config.enable_record_to_file(unique_dirname + '/' + 'depth_information.bag')

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().query_sensors()[1]
    depth_sensor.set_option(rs.option.enable_auto_exposure, False)
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            colorwriter.write(color_image)
            depthwriter.write(depth_colormap)

            cv2.imshow(previewName, depth_colormap)

            if cv2.waitKey(1) == ord("q") or time.time() - start_time >= 60:
                break
    finally:
        pipeline.stop()
        colorwriter.release()
        depthwriter.release()
        cv2.destroyWindow(previewName)


def cam_preview_thermal(previewName, camID):
    global thermowriter
    global thermowriter_raw
    global start_time
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
    try:
        while rval:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print(gray.shape)
            resized = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_NEAREST)
            thermowriter.write(resized)
            resized2 = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            thermowriter_raw.write(resized2)
            cv2.imshow(previewName, resized)
            rval, frame = cam.read()
            if cv2.waitKey(1) == ord("q") or time.time() - start_time >= 60:
                break
    finally:
        thermowriter.release()
        cv2.destroyWindow(previewName)


def create_new_project():
    global unique_dirname
    if not os.path.exists(unique_dirname):
        os.mkdir(unique_dirname)
    else:
        raise FileExistsError("[!] Program failed to generate a unique folder name. Try to re-run the program")


if __name__ == '__main__':
    create_new_project()

    colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 6, (640, 480), 1)
    depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 6, (640, 480), 1)
    thermowriter = cv2.VideoWriter(thermo_path, cv2.VideoWriter_fourcc(*'XVID'), 6, (640, 480), 0)
    thermowriter_raw = cv2.VideoWriter(thermo_raw_path, cv2.VideoWriter_fourcc(*'XVID'), 6, (640, 480), 1)
    # Need a solution for finding the camear and which porst they were initialized, check 0 if 1,2
    # Create two threads as follows
    thread1 = camThread("Camera 1", 1, "depth")
    thread2 = camThread("Camera 2", 2, "thermal")
    start_time = time.time()
    thread1.start()
    thread2.start()
