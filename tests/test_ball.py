import numpy as np
import pandas as pd

from ml.tracking.ball import reliable_ball, select_ball_path

GAP = range(40, 46)


def make_candidates(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(100):
        if f in GAP:
            continue
        x, y = 100 + 8 * f, 300 + 2 * f
        rows.append({"frame": f, "conf": 0.8, "x1": x - 8, "y1": y - 8, "x2": x + 8, "y2": y + 8})
        for _ in range(3):
            fx, fy = rng.uniform(0, 1000), rng.uniform(0, 600)
            rows.append({"frame": f, "conf": rng.uniform(0.05, 0.3), "x1": fx - 8, "y1": fy - 8, "x2": fx + 8, "y2": fy + 8})
    return pd.DataFrame(rows)


def test_path_follows_true_ball_and_interpolates_gap():
    traj = select_ball_path(make_candidates(), 100)
    true_x = 100 + 8 * np.arange(100)
    true_y = 300 + 2 * np.arange(100)
    assert np.abs(traj.cx.to_numpy() - true_x).max() < 3
    assert np.abs(traj.cy.to_numpy() - true_y).max() < 3
    assert traj.interpolated[list(GAP)].all()
    assert not traj.interpolated.drop(list(GAP)).any()


def test_reliable_ball_excludes_interpolated_and_low_conf():
    traj = select_ball_path(make_candidates(), 100)
    ok = reliable_ball(traj)
    assert not ok[list(GAP)].any()
    assert ok.sum() == 100 - len(GAP)


def test_no_candidates():
    empty = pd.DataFrame(columns=["frame", "conf", "x1", "y1", "x2", "y2"])
    traj = select_ball_path(empty, 10)
    assert len(traj) == 10 and traj.cx.isna().all()
