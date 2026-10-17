import cv2
import numpy as np

from ml.pitch.pitch_config import L, LINES, W
from ml.utils.video import iter_frames, to_h264, video_info

SCALE, PAD = 8, 20
TEAM_COLORS = {0: (0, 0, 255), 1: (255, 120, 0)}
REFEREE_COLOR = (0, 255, 255)
OTHER_COLOR = (160, 160, 160)
BALL_COLOR = (255, 0, 255)
REFEREE_CLASS = 3


def to_px(x, y):
    return int(PAD + x * SCALE), int(PAD + y * SCALE)


def pitch_canvas():
    img = np.full((int(W * SCALE) + 2 * PAD, int(L * SCALE) + 2 * PAD, 3), (40, 110, 40), np.uint8)
    for line in LINES:
        cv2.polylines(img, [np.array([to_px(x, y) for x, y in line], np.int32)], False, (255, 255, 255), 2)
    return img


def person_color(cls, team):
    if cls == REFEREE_CLASS:
        return REFEREE_COLOR
    return TEAM_COLORS.get(team, OTHER_COLOR) if team == team else OTHER_COLOR


def minimap_frame(canvas, players, ball=None):
    img = canvas.copy()
    for r in players.itertuples():
        color = person_color(r.cls, r.team)
        cv2.circle(img, to_px(r.X, r.Y), 7, color, -1)
        cv2.circle(img, to_px(r.X, r.Y), 7, (255, 255, 255), 1)
    if ball is not None:
        cv2.circle(img, to_px(*ball), 5, BALL_COLOR, -1)
    return img


def render_minimap(positions, ball_positions, n_frames, fps, out_path):
    canvas = pitch_canvas()
    h, w = canvas.shape[:2]
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame = {f: g for f, g in positions.groupby("frame")}
    empty = positions.iloc[0:0]
    for i in range(n_frames):
        writer.write(minimap_frame(canvas, by_frame.get(i, empty), ball_positions.get(i)))
    writer.release()
    to_h264(raw, out_path)
    raw.unlink()


def render_overlay(video, tracks, traj, out_path, trail=20):
    info = video_info(video)
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"], info["height"]))
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    points = []
    for i, frame in iter_frames(video):
        g = by_frame.get(i)
        if g is not None:
            for r in g.itertuples():
                color = person_color(r.cls_smooth, r.team)
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
                cv2.putText(frame, str(r.track_id), (int(r.x1), max(int(r.y1) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if i in traj.index and not np.isnan(traj.at[i, "cx"]):
            pt = (int(traj.at[i, "cx"]), int(traj.at[i, "cy"]))
            points = (points + [pt])[-trail:]
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(frame, a, b, BALL_COLOR, 2)
            cv2.circle(frame, pt, 10, BALL_COLOR, 2 if traj.at[i, "interpolated"] else -1)
        else:
            points = []
        writer.write(frame)
    writer.release()
    to_h264(raw, out_path)
    raw.unlink()
