"""Фигуры §7 → results/figures/*.png (matplotlib, 150 dpi, подписи на русском).

  python tools/make_figures.py            # все фигуры (EDA 1–5 из кэша/данных + результаты 6–12)
  python tools/make_figures.py --eda      # только 1–5
  python tools/make_figures.py --results  # только 6–12

Цвета классов — фиксированные слоты категориальной палитры (покой — синий, левая — оранжевый,
правая — бирюзовый), одинаковые во всех фигурах; расходящаяся шкала — синий↔красный с серой серединой.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, subject_of  # noqa: E402

FIG = ROOT / "results" / "figures"
EXP = ROOT / "results" / "experiments"
CLS_NAMES = ["покой", "левая", "правая"]
CLS_COL = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8f8e", "#a32b2b"])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False, "figure.facecolor": "white",
})


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("→", FIG / name)


# ============================================================ EDA (1–5)
def _montage() -> tuple[list[str], np.ndarray]:
    from scipy.io import loadmat
    m = loadmat(str(ROOT / "data" / "Montage.mat"), simplify_cells=True)["Montage"]
    return [str(x).lower() for x in m["EEG_labels"]], np.asarray(m["EEG_positions"], dtype=float)


def topo(ax, values: dict[str, float], vmax: float, title: str):
    """Топография по координатам Montage.mat (сопоставление по имени канала)."""
    from scipy.interpolate import griddata
    labels, pos = _montage()
    names = [n for n in values if n in labels and np.isfinite(values[n])]
    P = np.array([pos[labels.index(n)] for n in names])
    v = np.array([values[n] for n in names])
    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 120), np.linspace(-1.2, 1.2, 120))
    Z = griddata(P, v, (gx, gy), method="cubic")
    Z[np.hypot(gx, gy) > 1.08] = np.nan
    im = ax.contourf(gx, gy, Z, levels=np.linspace(-vmax, vmax, 21), cmap=DIV, extend="both")
    ax.add_patch(plt.Circle((0, 0), 1.08, fill=False, color=INK2, lw=1))
    ax.plot([-0.12, 0, 0.12], [1.06, 1.2, 1.06], color=INK2, lw=1)
    ax.scatter(P[:, 0], P[:, 1], s=6, color=INK, zorder=3)
    for n, (x, y) in zip(names, P):
        if n in ("c3", "c4", "cz"):
            ax.annotate(n.upper(), (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7, color=INK)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    return im


def eda_figures(E: dict) -> None:
    names = ["c4", "rpa", "f8", "p8", "f4", "p4", "fp2", "o2", "cz", "pz", "fz", "o1", "fp1", "p3", "f3", "p7", "f7",
             "lpa", "c3", "c7", "c8"]
    # 1. TFR
    tfr = np.nanmean(E["tfr"], 0)  # (cls, ch, f, t)
    f, t = E["tfr_f"], E["tfr_t"]
    fig, axs = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True, sharey=True)
    vmax = 2.5
    for i in range(3):
        for j, ch in enumerate(["C3", "Cz", "C4"]):
            ax = axs[i, j]
            im = ax.pcolormesh(t, f, tfr[i, j], cmap=DIV, vmin=-vmax, vmax=vmax, shading="auto")
            ax.axvline(0, color=INK, lw=1, ls="--")
            ax.grid(False)
            if i == 0:
                ax.set_title(ch)
            if j == 0:
                ax.set_ylabel(f"{CLS_NAMES[i]}\nчастота, Гц")
            if i == 2:
                ax.set_xlabel("время от начала сегмента, с")
    fig.colorbar(im, ax=axs, shrink=0.6, label="мощность относительно среднего покоя, дБ")
    fig.suptitle("ERD/ERS: время × частота, среднее по 56 сессиям (CAR по «чистым» каналам)", x=0.45)
    _save(fig, "erd_tfr_c3_c4_cz.png")

    # 2. топографии μ/β
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    for r, (key, band) in enumerate((("topo_mu", "μ 8–13 Гц"), ("topo_beta", "β 15–30 Гц"))):
        T = E[key]  # (sess, cls, ch) дБ
        contr = [("левая − покой", T[:, 1] - T[:, 0]), ("правая − покой", T[:, 2] - T[:, 0]),
                 ("левая − правая", T[:, 1] - T[:, 2])]
        for c, (title, D) in enumerate(contr):
            m = np.nanmean(D, 0)
            vals = {n: m[i] for i, n in enumerate(names) if n not in ("lpa", "rpa")}
            vmax = 3.0 if c < 2 else 0.4
            im = topo(axs[r, c], vals, vmax, f"{band}: {title}")
            fig.colorbar(im, ax=axs[r, c], shrink=0.7, label="дБ", ticks=[-vmax, 0, vmax])
    fig.suptitle("Топографии μ/β (окно 0.5–9 с сегмента), среднее по сессиям")
    _save(fig, "erd_topomaps.png")

    # 3. ФЭС
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    fr = E["fes_ratio"]
    ax = axs[0]
    q = np.nanpercentile(fr, [25, 50, 75], axis=0)
    ax.fill_between(E["fes_f"], q[0], q[2], color=SERIES[0], alpha=0.2, lw=0, label="межквартильный размах")
    ax.plot(E["fes_f"], q[1], color=SERIES[0], lw=1.5, label="медиана по сессиям")
    ax.axvspan(81.5, 85.5, color=SERIES[7], alpha=0.12, lw=0)
    ax.axvline(35, color=INK2, lw=1, ls=":")
    ax.text(36, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1, "граница признаков 35 Гц", fontsize=7, color=INK2)
    ax.annotate("пик ФЭС ≈ 83 Гц", (83.5, np.nanmax(q[1])), xytext=(95, np.nanmax(q[1])), fontsize=8,
                arrowprops={"arrowstyle": "-", "color": INK2}, color=INK)
    ax.set_xlim(1, 125)
    ax.set_xlabel("частота, Гц")
    ax.set_ylabel("воображение / покой, дБ (среднее по каналам)")
    ax.set_title("Спектр: воображение относительно покоя")
    ax.legend(loc="lower left", fontsize=7)
    ax = axs[1]
    tc = np.nanmean(E["fes_tc"], 0)
    for c in range(3):
        ax.plot(E["fes_tc_t"], tc[c], color=CLS_COL[c], label=CLS_NAMES[c])
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_xlabel("время от начала сегмента, с")
    ax.set_ylabel("мощность 81.5–85.5 Гц, дБ к покою")
    ax.set_title("Нарастание артефакта ФЭС внутри сегмента")
    ax.legend(fontsize=7)
    _save(fig, "fes_artifact_spectrum.png")

    # 4. ГЭОГ
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    H = E["heog"]  # (sess, cls, t)
    for c in range(3):
        m = H[:, c].mean(0)
        se = H[:, c].std(0) / np.sqrt(len(H))
        ax.fill_between(E["heog_t"], m - 1.96 * se, m + 1.96 * se, color=CLS_COL[c], alpha=0.15, lw=0)
        ax.plot(E["heog_t"], m, color=CLS_COL[c], label=CLS_NAMES[c])
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("время от начала сегмента, с")
    ax.set_ylabel("f7 − f8, 0.1–3 Гц (SD сессии)")
    ax.set_title("Горизонтальная ЭОГ (взгляд на стрелку) по классам, 95 % ДИ")
    ax.legend(fontsize=8)
    _save(fig, "heog_by_class.png")

    # 5. NIRS
    N = E["nirs"]  # (sess, Hb, cls, t, hemi)
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for h, hb in enumerate(["HbO", "HbR"]):
        for s, side in enumerate(["левое полушарие (x<0)", "правое полушарие (x>0)"]):
            ax = axs[h, s]
            for c in range(3):
                m = N[:, h, c, :, s].mean(0)
                se = N[:, h, c, :, s].std(0) / np.sqrt(len(N))
                ax.fill_between(E["nirs_t"], m - 1.96 * se, m + 1.96 * se, color=CLS_COL[c], alpha=0.15, lw=0)
                ax.plot(E["nirs_t"], m, color=CLS_COL[c], label=CLS_NAMES[c])
            ax.axvline(0, color=INK, lw=1, ls="--")
            ax.axvline(9, color=INK2, lw=1, ls=":")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{hb}, {side}")
            if s == 0:
                ax.set_ylabel("изменение, SD канала")
            if h == 1:
                ax.set_xlabel("время от начала сегмента, с")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"NIRS: grand average по классам ({len(N)} сессий с 12.5 Гц; базлайн −2…0 с)")
    _save(fig, "nirs_grand_average.png")


# ======================================================== результаты (6–12)


if __name__ == "__main__":
    if "--results" not in sys.argv:
        import eda_figures as ED
        eda_figures(ED.compute("--refresh" in sys.argv))
    if "--eda" not in sys.argv:
        from results_figures import SPEC, draw_all
        draw_all(SPEC)
