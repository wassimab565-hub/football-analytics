import threading
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.settings import Settings


def make_clip(path, seconds, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(int(seconds * fps)):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def settings(tmp_path, **overrides):
    weights = []
    for name in ("det.pt", "pitch.pt"):
        p = tmp_path / name
        p.write_bytes(b"x")
        weights.append(p)
    return Settings(detector_weights=weights[0], pitch_weights=weights[1], jobs_dir=tmp_path / "jobs", demo_json=tmp_path / "demo.json", **overrides)


def ok_runner(video, out_dir, progress):
    progress("tracking", 0.0)
    progress("ball", 0.5)
    (out_dir / "result.json").write_text('{"frames":[]}')


def upload(client, path, name="clip.mp4"):
    with open(path, "rb") as f:
        return client.post("/api/jobs", files={"file": (name, f, "video/mp4")})


def wait_for(client, job_id, status, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["status"] == status:
            return s
        time.sleep(0.05)
    raise AssertionError(f"job never reached {status}: {s}")


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    make_clip(p, 2)
    return p


def test_health_reports_live_when_weights_exist(tmp_path):
    client = TestClient(create_app(settings(tmp_path), ok_runner))
    assert client.get("/health").json() == {"status": "ok", "live": True}


def test_job_lifecycle(tmp_path, clip):
    client = TestClient(create_app(settings(tmp_path), ok_runner))
    r = upload(client, clip)
    assert r.status_code == 202
    job_id = r.json()["id"]
    status = wait_for(client, job_id, "done")
    assert status["progress"] == 1.0 and status["error"] is None
    assert client.get(f"/api/jobs/{job_id}/result").json() == {"frames": []}


def test_result_is_not_available_before_done(tmp_path, clip):
    gate = threading.Event()

    def slow(video, out_dir, progress):
        gate.wait(5)
        ok_runner(video, out_dir, progress)

    client = TestClient(create_app(settings(tmp_path), slow))
    job_id = upload(client, clip).json()["id"]
    wait_for(client, job_id, "running")
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409
    gate.set()
    wait_for(client, job_id, "done")


def test_second_job_waits_in_queue(tmp_path, clip):
    gate = threading.Event()

    def slow(video, out_dir, progress):
        gate.wait(5)
        ok_runner(video, out_dir, progress)

    client = TestClient(create_app(settings(tmp_path), slow))
    first = upload(client, clip).json()["id"]
    wait_for(client, first, "running")
    second = upload(client, clip).json()["id"]
    s = client.get(f"/api/jobs/{second}").json()
    assert s["status"] == "queued" and s["queue_position"] == 1
    gate.set()
    wait_for(client, second, "done")


def test_failed_job_reports_error(tmp_path, clip):
    def broken(video, out_dir, progress):
        raise RuntimeError("boom")

    client = TestClient(create_app(settings(tmp_path), broken))
    job_id = upload(client, clip).json()["id"]
    s = wait_for(client, job_id, "failed")
    assert "boom" in s["error"]
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


def test_rejects_long_video(tmp_path):
    long_clip = tmp_path / "long.mp4"
    make_clip(long_clip, 30, fps=1)
    client = TestClient(create_app(settings(tmp_path), ok_runner))
    r = upload(client, long_clip)
    assert r.status_code == 400 and "longer" in r.json()["detail"]
    assert not list((tmp_path / "jobs").iterdir())


def test_rejects_non_video_and_wrong_extension(tmp_path):
    client = TestClient(create_app(settings(tmp_path), ok_runner))
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a video")
    assert upload(client, fake).status_code == 400
    assert upload(client, fake, name="clip.avi").status_code == 400


def test_rejects_large_upload(tmp_path, clip):
    client = TestClient(create_app(settings(tmp_path, max_upload_mb=0), ok_runner))
    assert upload(client, clip).status_code == 413


def test_missing_weights_returns_503(tmp_path, clip):
    s = settings(tmp_path)
    s.detector_weights.unlink()
    client = TestClient(create_app(s, ok_runner))
    assert client.get("/health").json()["live"] is False
    assert upload(client, clip).status_code == 503


def test_unknown_job_and_demo(tmp_path):
    s = settings(tmp_path)
    client = TestClient(create_app(s, ok_runner))
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/demo").status_code == 404
    s.demo_json.write_text('{"fps":25}')
    assert client.get("/api/demo").json() == {"fps": 25}
