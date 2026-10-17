import pandas as pd

BALL_CLASS = 0
COLUMNS = ["frame", "conf", "x1", "y1", "x2", "y2"]


class Detector:
    def __init__(self, weights):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def ball_candidates(self, video, imgsz=1280, conf=0.05):
        rows = []
        for i, r in enumerate(self.model.predict(source=str(video), imgsz=imgsz, conf=conf, classes=[BALL_CLASS], stream=True, verbose=False)):
            for box, p in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                rows.append({"frame": i, "conf": p, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        return pd.DataFrame(rows, columns=COLUMNS)
