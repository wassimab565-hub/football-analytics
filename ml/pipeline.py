import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.detection.detector import Detector
from ml.export import export_json
from ml.pitch.homography import HomographyProvider, fit_homographies, to_pitch
from ml.pitch.keypoints import PitchKeypointModel
from ml.pitch.pitch_config import L, W
from ml.teams.team_classifier import assign_teams, track_features
from ml.tracking.ball import reliable_ball, select_ball_path
from ml.tracking.tracker import clean_tracks, track_people, write_tracker_config
from ml.utils.drawing import render_minimap
from ml.utils.video import to_gif, video_info

MARGIN = 5
STAGES = ("tracking", "teams", "ball", "pitch", "homography", "render")
SHARE = (0.42, 0.06, 0.14, 0.28, 0.02, 0.08)
START = dict(zip(STAGES, np.concatenate([[0.0], np.cumsum(SHARE)[:-1]])))


def inside_pitch(x, y):
    return (-MARGIN <= x) & (x <= L + MARGIN) & (-MARGIN <= y) & (y <= W + MARGIN)


def project_players(tracks, provider, smooth=5):
    parts = []
    for f, g in tracks.groupby("frame"):
        H = provider.get(int(f))
        if H is None:
            continue
        xy = to_pitch(H, np.column_stack([(g.x1 + g.x2) / 2, g.y2]))
        parts.append(pd.DataFrame({
            "frame": int(f),
            "track_id": g.track_id.to_numpy(),
            "cls": g.cls_smooth.to_numpy(),
            "team": g.team.to_numpy(),
            "X": xy[:, 0],
            "Y": xy[:, 1],
        }))
    if not parts:
        return pd.DataFrame(columns=["frame", "track_id", "cls", "team", "X", "Y"])
    proj = pd.concat(parts, ignore_index=True)
    proj = proj[inside_pitch(proj.X, proj.Y)].sort_values(["track_id", "frame"])
    for c in ("X", "Y"):
        proj[c] = proj.groupby("track_id")[c].transform(lambda s: s.rolling(smooth, center=True, min_periods=1).median())
    return proj


def project_ball(traj, provider):
    positions = {}
    for f in traj.index[reliable_ball(traj)]:
        H = provider.get(int(f))
        if H is None:
            continue
        x, y = to_pitch(H, [[traj.at[f, "cx"], traj.at[f, "cy"]]])[0]
        if inside_pitch(x, y):
            positions[int(f)] = (float(x), float(y))
    return positions


def run(video, detector_weights, pitch_weights, out_dir, progress=None, render_video=True):
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(video)

    def at(stage):
        if progress:
            progress(stage, float(START[stage]))

    detector = Detector(detector_weights)

    at("tracking")
    tracker_cfg = write_tracker_config(out_dir / "botsort.yaml")
    tracks = track_people(detector, video, tracker_cfg)
    tracks.to_csv(out_dir / "tracks.csv", index=False)

    at("teams")
    clean = clean_tracks(tracks)
    teams = assign_teams(track_features(video, clean))
    clean["team"] = clean.track_id.map(teams)
    clean.to_csv(out_dir / "tracks_teams.csv", index=False)

    at("ball")
    traj = select_ball_path(detector.ball_candidates(video), info["n_frames"])
    traj.to_csv(out_dir / "ball_track.csv")

    at("pitch")
    kps = PitchKeypointModel(pitch_weights).predict_video(video)
    kps.to_csv(out_dir / "pitch_keypoints.csv", index=False)

    at("homography")
    Hs, _ = fit_homographies(kps, info["width"], info["height"])
    provider = HomographyProvider(Hs)
    positions = project_players(clean, provider)
    positions.to_csv(out_dir / "minimap_positions.csv", index=False)
    ball = project_ball(traj, provider)
    export_json(positions, ball, info, out_dir / "result.json")

    if render_video:
        at("render")
        minimap = out_dir / f"{video.stem}_minimap.mp4"
        render_minimap(positions, ball, info["n_frames"], info["fps"], minimap)
        to_gif(minimap, out_dir / f"{video.stem}_minimap.gif")

    if progress:
        progress("done", 1.0)

    return {
        "frames": info["n_frames"],
        "tracks": int(clean.track_id.nunique()),
        "homography_frames": len(Hs),
        "projected_detections": len(positions),
        "ball_frames": len(ball),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--pitch-model", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(run(a.video, a.detector, a.pitch_model, a.out))


if __name__ == "__main__":
    main()
