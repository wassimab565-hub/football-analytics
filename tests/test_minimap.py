import shutil

import cv2
import numpy as np
import pandas as pd
import pytest

from ml.pitch.pitch_config import L, W
from ml.utils.drawing import TEAM_COLORS, minimap_frame, pitch_canvas, render_minimap, to_px


def positions():
    return pd.DataFrame({
        "frame": [0, 1], "track_id": [1, 1], "cls": [2, 2], "team": [0.0, 0.0], "X": [L / 2, L / 2 + 1], "Y": [W / 2, W / 2],
    })


def test_player_is_drawn_with_team_color():
    img = minimap_frame(pitch_canvas(), positions().iloc[[0]])
    x, y = to_px(L / 2, W / 2)
    assert tuple(int(v) for v in img[y, x]) == TEAM_COLORS[0]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_render_minimap_writes_video(tmp_path):
    out = tmp_path / "m.mp4"
    render_minimap(positions(), {0: (10.0, 10.0)}, 5, 10, out)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()
