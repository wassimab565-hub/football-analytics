import numpy as np
import pandas as pd


def select_ball_path(candidates, n_frames, miss=2.0, gap=12, v_max=150.0, v_scale=30.0):
    columns = ["cx", "cy", "conf", "interpolated"]
    if candidates.empty:
        return pd.DataFrame({"cx": np.nan, "cy": np.nan, "conf": np.nan, "interpolated": False}, index=range(n_frames))[columns]

    balls = candidates.sort_values("frame").reset_index(drop=True)
    balls["cx"] = (balls.x1 + balls.x2) / 2
    balls["cy"] = (balls.y1 + balls.y2) / 2

    fr = balls.frame.to_numpy()
    xy = balls[["cx", "cy"]].to_numpy()
    node = -np.log(balls.conf.to_numpy() + 1e-3)

    best = np.empty(len(balls))
    prev = np.full(len(balls), -1)
    lo = 0
    for i in range(len(balls)):
        best[i] = node[i] + miss * fr[i]
        while fr[lo] < fr[i] - gap:
            lo += 1
        for j in range(lo, i):
            dt = fr[i] - fr[j]
            if dt < 1:
                continue
            v = np.hypot(*(xy[i] - xy[j])) / dt
            if v > v_max:
                continue
            c = best[j] + node[i] + v / v_scale + miss * (dt - 1)
            if c < best[i]:
                best[i], prev[i] = c, j

    i = int(np.argmin(best + miss * (n_frames - 1 - fr)))
    path = []
    while i != -1:
        path.append(i)
        i = prev[i]
    path = path[::-1]

    sel = balls.loc[path, ["frame", "cx", "cy", "conf"]].set_index("frame")
    traj = sel.reindex(range(n_frames))
    inside = (traj.index >= sel.index.min()) & (traj.index <= sel.index.max())
    traj["interpolated"] = traj.cx.isna() & inside
    traj[["cx", "cy"]] = traj[["cx", "cy"]].interpolate(limit_area="inside")
    return traj[columns]


def reliable_ball(traj, min_conf=0.25):
    return traj.cx.notna() & ~traj.interpolated.astype(bool) & (traj.conf >= min_conf)
