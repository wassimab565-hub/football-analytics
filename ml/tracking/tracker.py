from pathlib import Path

import pandas as pd
import yaml

COLUMNS = ["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]
PEOPLE_CLASSES = (1, 2, 3)


def write_tracker_config(path, track_buffer=90):
    import ultralytics

    cfg = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/trackers/botsort.yaml").read_text())
    cfg["track_buffer"] = track_buffer
    Path(path).write_text(yaml.dump(cfg))
    return str(path)


def track_people(detector, video, tracker_cfg, conf=0.10, imgsz=1280, classes=PEOPLE_CLASSES):
    rows = []
    results = detector.new_model().track(
        source=str(video),
        imgsz=imgsz,
        conf=conf,
        classes=list(classes),
        tracker=tracker_cfg,
        agnostic_nms=True,
        stream=True,
        verbose=False,
    )
    for i, r in enumerate(results):
        if r.boxes.id is None:
            continue
        for box, tid, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            rows.append({"frame": i, "track_id": tid, "cls": c, "conf": p, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
    return pd.DataFrame(rows, columns=COLUMNS)


def clean_tracks(df, min_len=15, gk_share=None):
    length = df.groupby("track_id").frame.transform("size")
    df = df[length >= min_len].copy()
    if df.empty:
        df["cls_smooth"] = pd.Series(dtype=int)
        return df
    votes = df.groupby(["track_id", "cls"]).conf.sum().reset_index()
    best = votes.loc[votes.groupby("track_id").conf.idxmax()].set_index("track_id").cls
    if gk_share is not None:
        share = df.groupby("track_id").cls.apply(lambda s: (s == 1).mean())
        best[share[share >= gk_share].index] = 1
    df["cls_smooth"] = df.track_id.map(best)
    return df
