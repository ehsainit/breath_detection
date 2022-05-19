from src.yolo.yolov4 import YOLOv4


class Detector:
    def __init__(self, threshold=0.4, confidence=0.6):
        self.threshold = threshold
        self.confidence = confidence
        self.yolo = None

        self.init()

    def init(self):
        model = "src/yolo/models/yolov4-tiny-3l_best.weights"
        cfg = "src/yolo/models/yolov4-tiny-3l.cfg"

        self.yolo = YOLOv4(
            config=cfg,
            model=model,
            labels=["face"],
            confidence=self.confidence,
            threshold=self.threshold,
            use_gpu=False)

    def detect(self, frame):
        _, _, rt, results = self.yolo.inference(frame)
        if results:

            # sort by confidence
            results.sort(key=lambda e: e[0])
            best_detection = results[0]  # only one hand
            bb = list(best_detection[1:])

            return bb, rt
        return None, rt
