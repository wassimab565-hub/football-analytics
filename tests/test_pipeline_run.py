import json
import shutil

import cv2
import numpy as np
import pandas as pd
import pytest

import ml.pipeline as pipeline
from ml.pitch.pitch_config import PITCH

H_TRUE = np.array([[10.0, 2.0, 100.0], [0.5, 8.0, 60.0], [0.0003, 0.0008, 1.0]])
WIDTH, HEIGHT, N = 1400, 720, 20
WORLD = np.array([[25, 20], [40, 50], [55, 30], [65, 45], [80, 20], [90, 55]], np.float32)
COLORS = [(20, 20, 20)] * 3 + [(235, 235, 235)] * 3
BALL_WORLD = (50.0, 34.0)


def to_image(world):
    return cv2.perspectiveTransform(np.asarray(world, np.float32).reshape(-1, 1, 2), H_TRUE).reshape(-1, 2)


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (WIDTH, HEIGHT))
    img = to_image(WORLD)
    for _ in range(N):
        frame = np.full((HEIGHT, WIDTH, 3), (40, 140, 40), np.uint8)
        for (x, y), color in zip(img, COLORS):
            cv2.rectangle(frame, (int(x) - 15, int(y) - 60), (int(x) + 15, int(y)), color, -1)
        writer.write(frame)
    writer.release()


class FakeDetector:
    def __init__(self, weights):
        pass

    def ball_candidates(self, video):
        x, y = to_image([BALL_WORLD])[0]
        return pd.DataFrame({"frame": range(N), "conf": 0.9, "x1": x - 6, "y1": y - 6, "x2": x + 6, "y2": y + 6})


class FakePitchModel:
    def __init__(self, weights):
        pass

    def predict_video(self, video):
        img = to_image(PITCH)
        return pd.DataFrame([{"frame": f, "kp": k, "x": img[k, 0], "y": img[k, 1], "conf": 0.9} for f in range(N) for k in range(32)])


def fake_track_people(detector, video, cfg):
    img = to_image(WORLD)
    rows = [
        {"frame": f, "track_id": i + 1, "cls": 2, "conf": 0.9, "x1": x - 15, "y1": y - 60, "x2": x + 15, "y2": y}
        for f in range(N) for i, (x, y) in enumerate(img)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "Detector", FakeDetector)
    monkeypatch.setattr(pipeline, "PitchKeypointModel", FakePitchModel)
    monkeypatch.setattr(pipeline, "track_people", fake_track_people)
    monkeypatch.setattr(pipeline, "write_tracker_config", lambda path: str(path))
    video = tmp_path / "clip.mp4"
    make_video(video)
    return video, tmp_path / "out"


def test_run_end_to_end_with_fake_models(patched):
    video, out = patched
    calls = []
    summary = pipeline.run(video, "d.pt", "p.pt", out, progress=lambda s, f: calls.append((s, f)), render_video=False)

    assert summary["tracks"] == 6
    assert summary["homography_frames"] == N
    assert summary["projected_detections"] == 6 * N
    assert summary["ball_frames"] == N

    for name in ("tracks.csv", "tracks_teams.csv", "ball_track.csv", "pitch_keypoints.csv", "minimap_positions.csv", "result.json"):
        assert (out / name).exists()

    stages = [s for s, _ in calls]
    fractions = [f for _, f in calls]
    assert stages == ["tracking", "teams", "ball", "pitch", "homography", "done"]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0

    data = json.loads((out / "result.json").read_text())
    assert len(data["frames"]) == N
    first = {p[0]: p for p in data["frames"][0]["players"]}
    for i, (x, y) in enumerate(WORLD):
        assert abs(first[i + 1][3] - x) < 0.2 and abs(first[i + 1][4] - y) < 0.2
    assert len({first[i][1] for i in (1, 2, 3)}) == 1
    assert len({first[i][1] for i in (4, 5, 6)}) == 1
    assert first[1][1] != first[4][1]
    assert data["frames"][0]["ball"] is not None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_run_renders_minimap_files(patched):
    video, out = patched
    pipeline.run(video, "d.pt", "p.pt", out)
    assert (out / "clip_minimap.mp4").exists() and (out / "clip_minimap.gif").exists()
