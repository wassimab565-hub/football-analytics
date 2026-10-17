import numpy as np

L, W = 105.0, 68.0
PB_W, PB_L = 40.32, 16.5
GB_W, GB_L = 18.32, 5.5
SPOT, R = 11.0, 9.15

PITCH = np.array([
    (0, 0), (0, (W - PB_W) / 2), (0, (W - GB_W) / 2), (0, (W + GB_W) / 2), (0, (W + PB_W) / 2), (0, W),
    (GB_L, (W - GB_W) / 2), (GB_L, (W + GB_W) / 2),
    (SPOT, W / 2),
    (PB_L, (W - PB_W) / 2), (PB_L, (W - GB_W) / 2), (PB_L, (W + GB_W) / 2), (PB_L, (W + PB_W) / 2),
    (L / 2, 0), (L / 2, W / 2 - R), (L / 2, W / 2 + R), (L / 2, W),
    (L - PB_L, (W - PB_W) / 2), (L - PB_L, (W - GB_W) / 2), (L - PB_L, (W + GB_W) / 2), (L - PB_L, (W + PB_W) / 2),
    (L - SPOT, W / 2),
    (L - GB_L, (W - GB_W) / 2), (L - GB_L, (W + GB_W) / 2),
    (L, 0), (L, (W - PB_W) / 2), (L, (W - GB_W) / 2), (L, (W + GB_W) / 2), (L, (W + PB_W) / 2), (L, W),
    (L / 2 - R, W / 2), (L / 2 + R, W / 2),
], dtype=np.float32)

_t = np.linspace(0, 2 * np.pi, 60)
LINES = [
    [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],
    [(L / 2, 0), (L / 2, W)],
    [(0, (W - PB_W) / 2), (PB_L, (W - PB_W) / 2), (PB_L, (W + PB_W) / 2), (0, (W + PB_W) / 2)],
    [(L, (W - PB_W) / 2), (L - PB_L, (W - PB_W) / 2), (L - PB_L, (W + PB_W) / 2), (L, (W + PB_W) / 2)],
    [(0, (W - GB_W) / 2), (GB_L, (W - GB_W) / 2), (GB_L, (W + GB_W) / 2), (0, (W + GB_W) / 2)],
    [(L, (W - GB_W) / 2), (L - GB_L, (W - GB_W) / 2), (L - GB_L, (W + GB_W) / 2), (L, (W + GB_W) / 2)],
    list(zip(L / 2 + R * np.cos(_t), W / 2 + R * np.sin(_t))),
]

EXCLUDED_KEYPOINTS = (3, 9, 10, 11, 12, 17, 18, 19, 20, 28)
