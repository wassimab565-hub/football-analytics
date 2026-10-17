import cv2
import numpy as np
import pandas as pd

from ml.pipeline import project_players
from ml.pitch.homography import HomographyProvider, fit_homographies, to_pitch
from ml.pitch.pitch_config import PITCH

H_TRUE = np.array([[10.0, 2.0, 100.0], [0.5, 8.0, 60.0], [0.0003, 0.0008, 1.0]])
WIDTH, HEIGHT = 1400, 720


def to_image(world):
    return cv2.perspectiveTransform(np.asarray(world, np.float32).reshape(-1, 1, 2), H_TRUE).reshape(-1, 2)


def make_keypoints(frames=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(frames):
        img = to_image(PITCH) + rng.normal(0, 0.5, (32, 2))
        img[[1, 6]] = rng.uniform([50, 50], [1300, 650], (2, 2))
        rows.extend({"frame": f, "kp": k, "x": img[k, 0], "y": img[k, 1], "conf": 0.9} for k in range(32))
    return pd.DataFrame(rows)


def test_homography_recovers_pitch_despite_outliers():
    Hs, stats = fit_homographies(make_keypoints(), WIDTH, HEIGHT)
    assert len(Hs) == 5
    assert (stats.inliers >= 15).all()
    probe = np.array([[52.5, 34.0], [20.0, 10.0], [90.0, 60.0]], np.float32)
    back = to_pitch(Hs[0], to_image(probe))
    assert np.abs(back - probe).max() < 0.3


def test_provider_fills_short_gaps_only():
    Hs = {10: np.eye(3), 20: 2 * np.eye(3)}
    p = HomographyProvider(Hs, max_gap=10)
    assert p.get(15) is Hs[10]
    assert p.get(25) is Hs[20]
    assert p.get(5) is None
    assert p.get(40) is None


def test_project_players_matches_ground_truth():
    world = np.array([[30.0, 20.0], [60.0, 40.0]], np.float32)
    img = to_image(world)
    tracks = pd.DataFrame({
        "frame": [0, 0], "track_id": [1, 2], "cls_smooth": [2, 3], "team": [0.0, np.nan],
        "x1": img[:, 0] - 10, "x2": img[:, 0] + 10, "y1": img[:, 1] - 60, "y2": img[:, 1],
    })
    out = project_players(tracks, HomographyProvider({0: H_TRUE}))
    assert np.allclose(out[["X", "Y"]].to_numpy(), world, atol=0.05)
    assert out.team.isna().sum() == 1
