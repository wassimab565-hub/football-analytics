import json
from pathlib import Path

import pandas as pd

from ml.pitch.pitch_config import L, W


def export_json(positions, ball_positions, info, path):
    by_frame = {f: g for f, g in positions.groupby("frame")}
    frames = []
    for i in range(info["n_frames"]):
        g = by_frame.get(i)
        players = [] if g is None else [
            [int(r.track_id), -1 if pd.isna(r.team) else int(r.team), int(r.cls), round(float(r.X), 1), round(float(r.Y), 1)]
            for r in g.itertuples()
        ]
        b = ball_positions.get(i)
        frames.append({"players": players, "ball": None if b is None else [round(b[0], 1), round(b[1], 1)]})
    data = {
        "fps": round(float(info["fps"]), 3),
        "n_frames": info["n_frames"],
        "pitch": {"length": L, "width": W},
        "frames": frames,
    }
    Path(path).write_text(json.dumps(data, separators=(",", ":")))
    return path
