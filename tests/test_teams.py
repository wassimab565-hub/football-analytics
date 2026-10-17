import cv2
import numpy as np
import pandas as pd

from ml.teams.team_classifier import assign_teams, track_features

BLACK, WHITE = (20, 20, 20), (235, 235, 235)


def make_video(path, n=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    for _ in range(n):
        frame = np.full((360, 640, 3), (40, 140, 40), np.uint8)
        for i, color in enumerate([BLACK, BLACK, BLACK, WHITE, WHITE, WHITE]):
            x = 40 + i * 100
            cv2.rectangle(frame, (x, 100), (x + 40, 200), color, -1)
        writer.write(frame)
    writer.release()


def test_teams_separate_dark_and_light_shirts(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    rows = [
        {"frame": f, "track_id": i, "cls_smooth": 2, "conf": 0.9, "x1": 40 + i * 100, "y1": 100, "x2": 80 + i * 100, "y2": 200}
        for f in range(12) for i in range(6)
    ]
    feats = track_features(video, pd.DataFrame(rows))
    teams = assign_teams(feats)
    assert len(teams) == 6
    assert teams[[0, 1, 2]].nunique() == 1
    assert teams[[3, 4, 5]].nunique() == 1
    assert teams[0] != teams[3]
