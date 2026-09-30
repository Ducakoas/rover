from __future__ import annotations

from pathlib import Path
import csv
import numpy as np


def _s_curve() -> np.ndarray:
    x = np.linspace(-12.0, 12.0, 27)
    y = 1.75 * np.sin((x + 12.0) * np.pi / 12.0) * 0.72 + 0.42 * np.sin((x + 12.0) * np.pi / 4.8)
    y *= 0.62
    return np.column_stack([x, y])


def _gentle() -> np.ndarray:
    x = np.linspace(-12.0, 12.0, 23)
    y = 0.95 * np.sin((x + 12.0) * np.pi / 13.5)
    return np.column_stack([x, y])


def _hill() -> np.ndarray:
    x = np.linspace(-12.5, 12.5, 26)
    y = 0.55 * np.sin((x + 12.5) * np.pi / 8.0)
    return np.column_stack([x, y])


def _straight() -> np.ndarray:
    x = np.linspace(-12.0, 12.0, 26)
    y = np.zeros_like(x)
    return np.column_stack([x, y])


def _wide_curve() -> np.ndarray:
    x = np.linspace(-12.0, 12.0, 31)
    y = 2.25 * np.sin((x + 12.0) * np.pi / 24.0)
    return np.column_stack([x, y])


def resample_path(path_xy: np.ndarray, spacing: float = 0.35) -> np.ndarray:
    p = np.asarray(path_xy, dtype=float)
    if len(p) < 2:
        raise ValueError("A path needs at least two points")
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        raise ValueError("Path length is zero")
    sq = np.arange(0.0, s[-1] + 0.5 * spacing, spacing)
    if sq[-1] < s[-1]:
        sq = np.append(sq, s[-1])
    else:
        sq[-1] = s[-1]
    xq = np.interp(sq, s, p[:, 0])
    yq = np.interp(sq, s, p[:, 1])
    return np.column_stack([xq, yq])


def get_path(name: str) -> np.ndarray:
    name = name.lower()
    if name == "s_curve":
        return resample_path(_s_curve())
    if name == "gentle":
        return resample_path(_gentle())
    if name == "hill":
        return resample_path(_hill())
    if name == "straight":
        return resample_path(_straight())
    if name == "wide_curve":
        return resample_path(_wide_curve())
    raise KeyError(f"Unknown path: {name}")


def load_csv_path(path: str | Path) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"x", "y"}.issubset(set(h.lower() for h in reader.fieldnames)):
            raise ValueError("CSV path must contain columns x,y")
        hx = next(h for h in reader.fieldnames if h.lower() == "x")
        hy = next(h for h in reader.fieldnames if h.lower() == "y")
        for r in reader:
            rows.append([float(r[hx]), float(r[hy])])
    return resample_path(np.asarray(rows, dtype=float))
