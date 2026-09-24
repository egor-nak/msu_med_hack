"""Офлайн-обучение глобальных компонентов → artifacts/ (судья не запускает).

  python solution/train.py --mode all  --data data/train --out solution/artifacts
  python solution/train.py --mode loso --data data/train --out solution/artifacts   # → artifacts/loso/<SUBJ>/

Глобальная ЭЭГ-модель (межсубъектный перенос, B5):
  * каждая сессия прогоняется через ТЕ ЖЕ причинные фильтры, что и онлайн (hbci.eeg.EEGFrontEnd;
    подача порциями инвариантна к разбиению — никакого filtfilt);
  * плохие каналы определяются по данным блока 1 (как онлайн при первом fit_block) и интерполируются;
  * Cref_session — среднее ковариаций всех окон блока 1 (так же, как онлайн после блока 1);
  * признаки — касательное пространство в Cref_session для всех размеченных окон (с train_skip);
  * объединение по сессиям → стандартизация + LogisticRegression (class_weight='balanced').
Сохраняется global_eeg.joblib (numpy-коэффициенты + метаданные: train_files, конфигурация, версии).
Конфигурация ЭЭГ: artifacts/config.json (финальная) либо --cfg (JSON относительно DEFAULTS, как OC_CFG).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import io_utils as iu  # noqa: E402
from hbci.config import Config, resolve_channels  # noqa: E402
from hbci.eeg import EEGFrontEnd, spatial_matrix, transform  # noqa: E402
from hbci.pipeline import ts_vector  # noqa: E402
from hbci.riemann import estimate_cov, invsqrtm_spd, mean_spd  # noqa: E402

CACHE = HERE.parent / "results" / "cache" / "global_feats"
EEG_KEYS = ("channels", "reference", "hp_hz", "bands", "band_order", "cov", "cov_eps", "cov_avg_n", "mean", "train_skip_s",
            "bad_channels", "bad_flat_frac", "bad_ratio")


def session_features(path: Path, e: dict) -> tuple[np.ndarray, np.ndarray]:
    """TS-признаки и метки всех размеченных окон сессии (перецентрирование по блоку 1)."""
    s = iu.load_session(path)
    fs = s["fs_eeg"]
    win, step = int(round(fs)), int(round(fs * 0.25))
    fe = EEGFrontEnd(dict(e, features="ts"), fs, iu.EEG_CHANNEL_NAMES, len(s["states"]), win, step)
    st, bl = s["states"], s["blocks"]
    b1 = int(np.flatnonzero(bl == 1)[-1]) + 1
    fe.push(s["eeg"][:b1])
    bad = fe.bad_channels()
    K1 = fe.K
    fe.push(s["eeg"][b1:])
    M, _ = spatial_matrix(fe.names, fe.channels, e["reference"], bad, interpolate=True)
    n = int(e.get("cov_avg_n", 1))
    S = fe.S[: fe.K]
    if n > 1:  # как HybridDecoder._S: среднее по окнам k−n+1 … k
        kk = np.arange(fe.K)
        S = sum(fe.S[np.maximum(kk - j, 0)] for j in range(n)) / n
    C = estimate_cov(transform(S, M), win, e["cov"], e["cov_eps"])
    isq = invsqrtm_spd(mean_spd(C[:K1], e["mean"], 400))
    skip = int(round(e["train_skip_s"] * fs))
    ks, ys = [], []
    d = np.flatnonzero(np.diff(np.r_[-1, st, -1]) != 0)
    for a, b in zip(d[:-1], d[1:]):
        c = int(st[a])
        if c not in (1, 2, 3):
            continue
        for k in range(-(-(a + skip) // step), (b - win) // step + 1):
            if k < fe.K and bl[fe.ends[k] - 1] == bl[a]:
                ks.append(k)
                ys.append(c)
    ks = np.asarray(ks)
    return ts_vector(isq @ C[ks] @ isq).astype(np.float32), np.asarray(ys, dtype=np.int8)


def cached_features(path: Path, e: dict) -> tuple[np.ndarray, np.ndarray]:
    key = hashlib.md5(json.dumps({k: e[k] for k in EEG_KEYS}, sort_keys=True).encode()).hexdigest()[:10]
    p = CACHE / key / f"{path.stem}.npz"
    if p.exists():
        z = np.load(p)
        return z["X"], z["y"]
    X, y = session_features(path, e)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=X, y=y)
    return X, y


def fit_global(files: list[Path], e: dict, C: float, feats: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    import scipy
    import sklearn
    X = np.concatenate([feats[f.stem][0] for f in files]).astype(np.float64)
    y = np.concatenate([feats[f.stem][1] for f in files]).astype(int)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=500, random_state=0).fit((X - mu) / sd, y)
    return {
        "meta": {"eeg": {k: e[k] for k in EEG_KEYS}, "channels": resolve_channels(e["channels"]),
                 "train_files": [f.name for f in files], "C": C, "n_windows": int(len(y)),
                 "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}},
        "mu": mu, "sd": sd, "W": m.coef_, "b": m.intercept_,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "loso"], default="all")
    ap.add_argument("--data", default=str(HERE.parent / "data" / "train"))
    ap.add_argument("--out", default=str(HERE / "artifacts"))
    ap.add_argument("--cfg", default=None, help="JSON-переопределение конфигурации (как OC_CFG)")
    ap.add_argument("--C", type=float, default=None, help="C логистической регрессии (иначе global.C или 0.1)")
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args()
    # --cfg задаёт конфигурацию относительно DEFAULTS (как OC_CFG в экспериментах); без него — config.json
    cfg = Config.build(Path(a.out), env={"OC_CFG": a.cfg} if a.cfg else {})
    e = cfg.eeg
    C = a.C if a.C is not None else float(cfg.glob.get("C", 0.1))
    files = sorted(Path(a.data).glob("*.mat"))
    print(f"сессий: {len(files)}; каналы={e['channels']} ref={e['reference']} полосы={e['bands']} C={C}")
    from multiprocessing import Pool
    with Pool(a.j) as pool:
        res = pool.starmap(cached_features, [(f, e) for f in files])
    feats = {f.stem: r for f, r in zip(files, res)}
    out = Path(a.out)
    if a.mode == "all":
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit_global(files, e, C, feats), out / "global_eeg.joblib")
        print(f"→ {out / 'global_eeg.joblib'}")
    else:
        subs = sorted({f.stem.split("_")[-1] for f in files})
        for s in subs:
            tr = [f for f in files if f.stem.split("_")[-1] != s]
            d = out / "loso" / s
            d.mkdir(parents=True, exist_ok=True)
            joblib.dump(fit_global(tr, e, C, feats), d / "global_eeg.joblib")
            print(f"  LOSO {s}: {len(tr)} сессий → {d / 'global_eeg.joblib'}")


if __name__ == "__main__":
    main()
