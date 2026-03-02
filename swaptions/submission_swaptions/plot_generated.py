from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SURFACE_COL_RE = re.compile(r"^(Strike|Tenor)\s*:\s*([^;]+)\s*;\s*Maturity\s*:\s*(.+)$")


def _to_float_token(token: str) -> float:
    t = str(token).strip().upper()
    mult = 1.0
    if t.endswith("Y"):
        t = t[:-1]
    elif t.endswith("M"):
        t = t[:-1]
        mult = 1.0 / 12.0
    return float(t) * mult


def _parse_surface_columns(surface_cols: list[str]) -> tuple[list[float], list[float], dict[str, tuple[float, float]]]:
    parsed: dict[str, tuple[float, float]] = {}
    strikes: set[float] = set()
    maturities: set[float] = set()
    for col in surface_cols:
        m = SURFACE_COL_RE.match(col)
        if m is None:
            raise ValueError(f"Could not parse surface column: {col}")
        strike = _to_float_token(m.group(2))
        maturity = _to_float_token(m.group(3))
        parsed[col] = (strike, maturity)
        strikes.add(strike)
        maturities.add(maturity)
    return sorted(strikes), sorted(maturities), parsed


def _row_to_grid(
    row: pd.Series,
    surface_cols: list[str],
    col_meta: dict[str, tuple[float, float]],
    strikes: list[float],
    maturities: list[float],
) -> np.ndarray:
    s2i = {s: i for i, s in enumerate(strikes)}
    m2i = {m: j for j, m in enumerate(maturities)}
    grid = np.full((len(strikes), len(maturities)), np.nan, dtype=float)
    for col in surface_cols:
        s, m = col_meta[col]
        grid[s2i[s], m2i[m]] = float(row[col])
    return grid


def _plot_heatmap(
    grid: np.ndarray,
    strikes: list[float],
    maturities: list[float],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Tenor/Strike")
    ax.set_xticks(np.arange(len(maturities)))
    ax.set_xticklabels([f"{m:g}" for m in maturities], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(strikes)))
    ax.set_yticklabels([f"{s:g}" for s in strikes], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Implied Vol")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_atm_term_structure(
    df: pd.DataFrame,
    surface_cols: list[str],
    col_meta: dict[str, tuple[float, float]],
    strikes: list[float],
    maturities: list[float],
    tag: str,
    out_path: Path,
) -> None:
    atm_strike = min(strikes, key=lambda x: abs(x - 1.0))
    atm_cols = []
    for m in maturities:
        match = [c for c in surface_cols if col_meta[c] == (atm_strike, m)]
        if not match:
            raise RuntimeError(f"Missing ATM column for strike={atm_strike}, maturity={m}")
        atm_cols.append(match[0])

    vals = df[atm_cols].to_numpy(dtype=float)
    x = np.array(maturities, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(vals.shape[0]):
        alpha = 0.35
        lw = 1.0
        label = None
        if i == 0:
            alpha = 0.95
            lw = 2.0
            label = f"first ({df['Date'].iloc[i]})"
        elif i == vals.shape[0] - 1:
            alpha = 0.95
            lw = 2.0
            label = f"last ({df['Date'].iloc[i]})"
        ax.plot(x, vals[i], color="tab:blue", alpha=alpha, lw=lw, label=label)
    ax.set_title(f"{tag}: ATM (K={atm_strike:g}) term structure over horizon")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Implied Vol")
    ax.grid(alpha=0.2)
    if vals.shape[0] >= 2:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_smiles_first_last(
    df: pd.DataFrame,
    surface_cols: list[str],
    col_meta: dict[str, tuple[float, float]],
    strikes: list[float],
    maturities: list[float],
    tag: str,
    out_path: Path,
) -> None:
    pick = [0.25, 1.0, 5.0, 30.0]
    picked_mats = [min(maturities, key=lambda x: abs(x - p)) for p in pick]
    picked_mats = sorted(set(picked_mats))

    i0 = 0
    i1 = len(df) - 1
    fig, ax = plt.subplots(figsize=(9, 5))
    for mat in picked_mats:
        cols_for_mat = [(c, col_meta[c][0]) for c in surface_cols if abs(col_meta[c][1] - mat) < 1e-12]
        cols_for_mat = sorted(cols_for_mat, key=lambda item: item[1])
        strikes_m = [s for _, s in cols_for_mat]
        c_names = [c for c, _ in cols_for_mat]
        v0 = df.loc[i0, c_names].to_numpy(dtype=float)
        v1 = df.loc[i1, c_names].to_numpy(dtype=float)
        ax.plot(strikes_m, v0, lw=1.7, label=f"T={mat:g} first")
        ax.plot(strikes_m, v1, lw=1.7, ls="--", label=f"T={mat:g} last")

    ax.set_title(f"{tag}: smile slices (first vs last forecast date)")
    ax.set_xlabel("Tenor/Strike")
    ax.set_ylabel("Implied Vol")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _make_plots_for_csv(csv_path: Path, out_dir: Path) -> tuple[str, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError(f"Date column missing in {csv_path}")
    surface_cols = [c for c in df.columns if c != "Date"]
    strikes, maturities, col_meta = _parse_surface_columns(surface_cols)

    tag = csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    g_first = _row_to_grid(df.iloc[0], surface_cols, col_meta, strikes, maturities)
    g_last = _row_to_grid(df.iloc[-1], surface_cols, col_meta, strikes, maturities)

    _plot_heatmap(
        g_first,
        strikes,
        maturities,
        title=f"{tag}: surface heatmap (first forecast date {df['Date'].iloc[0]})",
        out_path=out_dir / f"{tag}_surface_first.png",
    )
    _plot_heatmap(
        g_last,
        strikes,
        maturities,
        title=f"{tag}: surface heatmap (last forecast date {df['Date'].iloc[-1]})",
        out_path=out_dir / f"{tag}_surface_last.png",
    )
    _plot_atm_term_structure(
        df=df,
        surface_cols=surface_cols,
        col_meta=col_meta,
        strikes=strikes,
        maturities=maturities,
        tag=tag,
        out_path=out_dir / f"{tag}_atm_term_structure.png",
    )
    _plot_smiles_first_last(
        df=df,
        surface_cols=surface_cols,
        col_meta=col_meta,
        strikes=strikes,
        maturities=maturities,
        tag=tag,
        out_path=out_dir / f"{tag}_smiles_first_last.png",
    )
    return tag, g_first, g_last


def _plot_diff(a_tag: str, b_tag: str, a_grid: np.ndarray, b_grid: np.ndarray, out_path: Path, which: str) -> None:
    diff = b_grid - a_grid
    vmax = float(np.nanmax(np.abs(diff)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(diff, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{b_tag} - {a_tag}: surface diff ({which})")
    ax.set_xlabel("Maturity index")
    ax.set_ylabel("Strike index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta implied vol")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots from generated submission CSV files.")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to generated CSV. Pass multiple --csv flags to plot multiple outputs.",
    )
    parser.add_argument("--out_dir", type=str, default="results/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    produced: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in args.csv:
        csv_path = Path(p)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        produced.append(_make_plots_for_csv(csv_path, out_dir))

    if len(produced) >= 2:
        a_tag, a_first, a_last = produced[0]
        b_tag, b_first, b_last = produced[1]
        _plot_diff(a_tag, b_tag, a_first, b_first, out_dir / f"{b_tag}_minus_{a_tag}_diff_first.png", "first")
        _plot_diff(a_tag, b_tag, a_last, b_last, out_dir / f"{b_tag}_minus_{a_tag}_diff_last.png", "last")

    print(f"Saved plots under: {out_dir}")
    for file in sorted(out_dir.glob("*.png")):
        print(file)


if __name__ == "__main__":
    main()
