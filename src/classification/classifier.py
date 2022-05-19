import numpy as np
import math
import cv2
from statistics import mode

from utils import WIDTH, HEIGHT


class Classifier:
    def __int__(self):
        pass

    def determine_face_heat(self):
        pass

    def determine_roi_depth(self, color_frame, roi, depth_mask, debug=False):
        xt, yt, xb, yb = roi

        p_min = (xt, yt)
        p_max = (xb, yb)
        cx = xt + xb // 2
        cy = yt + yb // 2
        tmp = []

        # there is a faster iteration method using cpython
        for y in range(p_min[1], p_max[1]):
            for x in range(p_min[0], p_max[0]):
                if 0 < x < WIDTH and 0 < y < HEIGHT:
                    if depth_mask.get_distance(x, y) > 0:
                        tmp.append(depth_mask.get_distance(x, y))
        if debug:
            centriod = (cx, cy)
            text = "{:.2f}".format(mode(tmp))
            cv2.putText(color_frame, text, centriod, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        m = round(mode(tmp), 1)
        return m

    def classifiy(self):
        pass