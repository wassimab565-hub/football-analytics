import cv2
import pandas as pd

from ml.utils.video import iter_frames, video_info

INPUT_SIZE = 960
COLUMNS = ["frame", "kp", "x", "y", "conf"]


class PitchKeypointModel:
    def __init__(self, weights):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def predict_video(self, video, batch_size=16, conf=0.05):
        info = video_info(video)
        sx, sy = info["width"] / INPUT_SIZE, info["height"] / INPUT_SIZE
        rows, frames, ids = [], [], []

        def flush():
            for f, r in zip(ids, self.model.predict(frames, imgsz=INPUT_SIZE, conf=conf, verbose=False)):
                if r.keypoints is None or len(r.boxes) == 0:
                    continue
                j = int(r.boxes.conf.argmax())
                xy = r.keypoints.xy[j].cpu().numpy() * [sx, sy]
                cf = r.keypoints.conf[j].cpu().numpy()
                rows.extend({"frame": f, "kp": k, "x": float(xy[k, 0]), "y": float(xy[k, 1]), "conf": float(cf[k])} for k in range(len(xy)))
            frames.clear()
            ids.clear()

        for f, frame in iter_frames(video):
            frames.append(cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)))
            ids.append(f)
            if len(frames) == batch_size:
                flush()
        if frames:
            flush()
        return pd.DataFrame(rows, columns=COLUMNS)


def train_pitch_model(data, project, name="pitch_yolo26s_pose_960", model="yolo26s-pose.pt", epochs=100, imgsz=960, batch=8):
    from ultralytics import YOLO

    return YOLO(model).train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, seed=0, workers=2, project=str(project), name=name)
