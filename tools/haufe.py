"""§5.4 Паттерны Haufe: A = Σ_X·W·Σ_ŝ⁻¹.

1) log-power-вариант финальной конфигурации (те же каналы/референс/полосы, sLDA): для контрастов
   «левая − покой», «правая − покой», «левая − правая» обучается бинарный sLDA на всех окнах сессии
   (офлайн-анализ), паттерн A = cov(X)·w / var(Xw) в единицах лог-мощности; нормировка на max|A| по сессии,
   среднее по сессиям → топографии μ (8–12 Гц) и β (16–24 Гц).
2) CSP-фильтры B2 (полоса 7–13 Гц, all17): паттерн фильтра с минимальным собственным числом для класса
   (сильнее всего подавленная мощность, ERD) — |A| по сессиям.
Выход: results/figures/haufe_patterns.png, results/experiments/E11_controls/haufe.json.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, segments_of  # noqa: E402

OUT = ROOT / "results" / "experiments" / "E11_controls"


def _cfg() -> dict:
    p = ROOT / "solution" / "artifacts" / "config.json"
    sys.path.insert(0, str(ROOT / "solution"))
    from hbci.config import Config
    return Config.build(ROOT / "solution" / "artifacts", env={}).eeg if p.exists() else Config.build(None, env={}).eeg


def _session(path: str, e: dict, bands: list, channels, reference: str, csp: bool) -> dict:
    sys.path.insert(0, str(ROOT / "solution"))
    import io_utils as iu
    from hbci.eeg import EEGFrontEnd, spatial_matrix, transform
    from hbci.models import fit_csp
    from hbci.riemann import estimate_cov
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    s = iu.load_session(path)
    cfg = dict(e, bands=bands, channels=channels, reference=reference, features="logpower")
    fe = EEGFrontEnd(cfg, 250.0, iu.EEG_CHANNEL_NAMES, len(s["states"]), 250, 62)
    fe.push(s["eeg"])
    M, ch = spatial_matrix(fe.names, fe.channels, reference, fe.bad_channels(), interpolate=True)
    C = transform(fe.S[: fe.K], M)
    ks, ys = [], []
    for g in segments_of(s["states"], s["blocks"]):
        for k in range(-(-g["start"] // 62), (g["end"] - 250) // 62 + 1):
            if k < fe.K:
                ks.append(k)
                ys.append(g["cls"])
    ks, ys = np.array(ks), np.array(ys)
    out = {"channels": ch}
    if not csp:
        X = np.log(np.diagonal(C[ks], axis1=-2, axis2=-1) + 1e-12).reshape(len(ks), -1)
        for name, (a, b) in {"L-rest": (1, 2), "R-rest": (1, 3), "L-R": (3, 2)}.items():
            m = np.isin(ys, [a, b])
            Xm, yy = X[m], (ys[m] == b).astype(int)  # положительный паттерн — признак выше у b
            w = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xm, yy).coef_[0]
            Xc = Xm - Xm.mean(0)
            s_hat = Xc @ w
            A = (Xc.T @ s_hat / len(s_hat)) / s_hat.var()
            A = A * np.sign(np.mean(Xm[yy == 1] @ w) - np.mean(Xm[yy == 0] @ w))
            out[name] = (A / np.abs(A).max()).reshape(len(bands), -1)
    else:
        Cs = estimate_cov(C[ks], 250, "lw", 1e-4)
        W = fit_csp(Cs, ys, 2)  # (B, p, 6): на класс [max, min]
        b = 1  # 7–13 Гц
        Cb = Cs[:, b].mean(0)
        for ci, name in ((1, "L"), (2, "R")):
            wv = W[b][:, 2 * ci + 1]      # фильтр min λ класса (ERD)
            A = Cb @ wv / (wv @ Cb @ wv)
            out[name] = np.abs(A) / np.abs(A).max()
    return out


def main() -> None:
    import make_figures as MF
    plt = MF.plt
    e = _cfg()
    files = [str(f) for f in sorted(DATA.glob("*.mat"))]
    bands = e["bands"]
    with Pool(8) as pool:
        R = pool.starmap(_session, [(f, e, bands, e["channels"], e["reference"], False) for f in files])
        Rc = pool.starmap(_session, [(f, e, [[4, 7], [7, 13], [13, 30]], "all17", "none", True) for f in files])
    ch = R[0]["channels"]
    bands_f = [tuple(b) for b in bands]
    mu = [i for i, b in enumerate(bands_f) if b[0] >= 8 and b[1] <= 12]
    beta = [i for i, b in enumerate(bands_f) if b[0] >= 16 and b[1] <= 24]
    res: dict = {"channels": ch, "bands": bands}
    fig, axs = plt.subplots(3, 3, figsize=(11, 10))
    for c, name in enumerate(["L-rest", "R-rest", "L-R"]):
        P = np.mean([r[name] for r in R], 0)  # (B, ch)
        for r_i, (lab, idx) in enumerate((("μ 8–12 Гц", mu), ("β 16–24 Гц", beta))):
            v = P[idx].mean(0)
            res[f"{name}:{lab}"] = dict(zip(ch, np.round(v, 3).tolist()))
            ttl = {"L-rest": "левая − покой", "R-rest": "правая − покой", "L-R": "левая − правая"}[name]
            vm = float(np.abs(v).max())
            im = MF.topo(axs[r_i, c], dict(zip(ch, v)), vm, f"{lab}: {ttl}")
            fig.colorbar(im, ax=axs[r_i, c], shrink=0.6, label="паттерн (норм.)", ticks=[-vm, 0, vm], format="%.2f")
    for c, name in enumerate(["L", "R"]):
        chc = Rc[0]["channels"]
        v = np.mean([r[name] for r in Rc], 0)
        res[f"CSP_{name}"] = dict(zip(chc, np.round(v, 3).tolist()))
        vm = float(np.abs(v - v.mean()).max())
        im = MF.topo(axs[2, c], dict(zip(chc, v - v.mean())), vm,
                     f"B2 CSP 7–13 Гц: {'левая' if name == 'L' else 'правая'}\n(фильтр ERD)")
        fig.colorbar(im, ax=axs[2, c], shrink=0.6, label="|A| − среднее", ticks=[-vm, 0, vm], format="%.2f")
    axs[2, 2].axis("off")
    axs[2, 2].text(0.08, 0.5, "Паттерны Haufe: A = Σ_X·w / var(ŝ)\nlog-power + sLDA, sensorimotor11\n\n"
                   "контраст «X − Y»: отрицательные\nзначения — мощность у X ниже, чем у Y\n(для «левая − покой» — ERD при левой)",
                   fontsize=8, va="center")
    fig.suptitle("Паттерны Haufe: где модель «видит» различия классов (среднее по 56 сессиям)")
    MF._save(fig, "haufe_patterns.png")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "haufe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    lr = res["L-R:μ 8–12 Гц"]
    print("μ, левая−правая: C3 =", lr.get("c3"), " C4 =", lr.get("c4"))


if __name__ == "__main__":
    main()
