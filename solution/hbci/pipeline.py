"""HybridDecoder — онлайн-декодер: push / predict / fit_block.

Поток данных:
  push      → EEGFrontEnd (ковариации окон на сетке) и NIRSFrontEnd (отфильтрованные HbO/HbR);
  fit_block → разметка прошедшего блока → окна сетки, целиком лежащие в сегментах;
              маска плохих каналов и референс (по прошлым данным); сессионная ЭЭГ-модель
              (TS / MDM / log-power / CSP + sLDA/LR); глобальная модель (перецентрирование по блоку 1);
              NIRS-модель; калибровка смещений под min-F1;
  predict   → лог-вероятности ЭЭГ (смесь сессионной и глобальной) и NIRS → EMA по вызовам →
              слияние → смещение → argmax.

Причинность и тайминг:
  * метки используются только в fit_block и только для прошедших блоков;
  * центры, нормировки, маски каналов, α/β NIRS — по уже полученным данным;
  * ответ не зависит от размеров порций push и пауз между вызовами (EMA по числу вызовов).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import Config
from .csp import CSP_CHANNELS, CSPArtifact, features_from_cov
from .eeg import EEGFrontEnd, spatial_matrix, transform
from .fusion import calibrate_bias, fuse
from .models import CLASSES, GlobalEEG, csp_features, fit_csp, fit_linear, fit_mdm, fit_ts_mdm, log_softmax
from .nirs import NIRSFrontEnd
from .riemann import estimate_cov, expm_sym, invsqrtm_spd, logm_spd, mean_spd, triu_weights
from .smoothing import CallEMA, ema_sequence

UNIFORM = np.log(np.full(3, 1.0 / 3))


def ts_vector(Cw: np.ndarray) -> np.ndarray:
    """Касательный вектор перецентрированных ковариаций (N, B, p, p) → (N, B·p(p+1)/2)."""
    L = logm_spd(Cw)
    i, j, w = triu_weights(Cw.shape[-1])
    return (L[..., i, j] * w).reshape(len(Cw), -1)


class HybridDecoder:
    def __init__(self, meta: dict[str, Any], cfg: Config) -> None:
        self.meta = meta
        self.cfg = cfg
        fs = float(meta["fs_eeg"])
        self.fs = fs
        self.win = int(round(fs * 1.0))
        self.step = int(round(fs * 0.25))
        e = cfg.eeg
        self.fe = EEGFrontEnd(e, fs, list(meta["eeg_channel_names"]), int(meta.get("n_samples_eeg", 200_000)),
                              self.win, self.step)
        self.feat = e["features"]
        self.t = 0
        # NIRS
        self.meta_nirs = bool(meta.get("has_nirs")) and bool(meta.get("fs_nirs"))
        self.nirs_on = self.meta_nirs and bool(cfg.nirs["enabled"])
        self.ne = (NIRSFrontEnd(cfg.nirs, fs, float(meta["fs_nirs"]), int(meta.get("n_nirs_ch") or 0))
                   if self.nirs_on else None)
        # разметка прошедших блоков: точка сетки, класс, блок, «годится для обучения» (train_skip)
        self.ev_k: list[int] = []
        self.ev_y: list[int] = []
        self.ev_b: list[int] = []
        self.ev_tr: list[bool] = []
        self.n_blocks = 0
        # модели
        self.M, self.ch_sess = spatial_matrix(self.fe.names, self.fe.channels, e["reference"])
        self.cref_isqrt: np.ndarray | None = None
        self.m_lin = None
        self.m_mdm = None
        self.csp_W: np.ndarray | None = None
        self.m_nirs = None
        self.glob: GlobalEEG | None = None
        self.M_glob: np.ndarray | None = None
        self.glob_isqrt: np.ndarray | None = None
        self.fusion_file: dict[str, Any] = {}
        self._lcache: dict[tuple, dict] = {}
        # CSP + лог-мощность + асимметрия (фильтры обучены офлайн, здесь только применяются)
        self.csp_cfg = e.get("csp") or {}
        self.csp: CSPArtifact | None = None
        self.M_csp: np.ndarray | None = None
        self.b_sess: dict[str, np.ndarray] = {}
        self.bias_cur = {k: np.asarray(cfg.bias["fixed"], dtype=float) for k in ("hyb", "eeg", "nirs")}
        sm = cfg.smoothing
        self.ema_eeg = CallEMA(sm["tau_s"], sm["calls_per_s"], sm["space"])
        tn = sm["tau_s"] if sm.get("nirs_tau_s") is None else sm["nirs_tau_s"]
        self.ema_nirs = CallEMA(tn, sm["calls_per_s"], sm["space"])
        self.ema_a = self.ema_eeg.a
        self.ema_an = self.ema_nirs.a
        # дамп вероятностей (только эксперименты)
        self.dump_dir = os.environ.get("HBCI_DUMP_DIR")
        self.dump_rows: list[list[float]] = []
        self.dump_started = False

    # ================================================================== io
    def load(self, artifacts_dir: str | Path | None) -> None:
        """Глобальные компоненты: artifacts/ (по умолчанию) или HBCI_LOSO_DIR/<subject>/ (эксперименты)."""
        loso = os.environ.get("HBCI_LOSO_DIR")
        if loso:
            d: Path | None = Path(loso) / str(self.meta["subject"])
        elif self.cfg.raw.get("_use_artifacts", True) and artifacts_dir is not None:
            d = Path(artifacts_dir)
        else:
            d = None
        if d is not None and self.cfg.glob["enabled"]:
            p = d / self.cfg.glob["file"]
            if p.exists():
                self.glob = GlobalEEG.from_dict(joblib.load(p))
                self.glob.check_compatible(self.cfg.eeg, self.fe.channels)
        if self.csp_cfg.get("enabled"):
            if not self.csp_cfg.get("use_csp", True):
                # только лог-мощность/асимметрия: CSP-файл не нужен, каналы — порядок chan_dict
                self.csp = CSPArtifact(np.zeros((0, len(CSP_CHANNELS))), list(CSP_CHANNELS), self.csp_cfg["prep"],
                                       {"note": "без CSP-фильтров"})
            else:
                p = (d / self.csp_cfg["file"]) if d is not None else None
                if p is None or not p.exists():
                    raise FileNotFoundError(f"eeg.csp.enabled, но нет файла CSP: {p}")
                self.csp = CSPArtifact.from_dict(joblib.load(p))
            self.csp.check(self.fe.names, self.csp_cfg["prep"])
        if d is not None and self.cfg.fusion["use_file"]:
            p = d / self.cfg.fusion["file"]
            if p.exists():
                self.fusion_file = json.loads(p.read_text(encoding="utf-8"))
        self._update_bias()
        self._warmup()

    def _warmup(self) -> None:
        """Прогрев ленивых инициализаций (первый fit_block/predict не платит за них)."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))
        y = np.repeat([1, 2, 3], 10)
        fit_linear(X, y, "slda")
        fit_linear(X, y, "lr")
        logm_spd(np.eye(3)[None] * 2.0)

    # ================================================================ push
    def push(self, eeg_chunk: np.ndarray, nirs_chunk: tuple | None) -> None:
        x = np.asarray(eeg_chunk)
        self.fe.push(x)
        self.t += x.shape[0]
        if self.ne is not None and nirs_chunk is not None:
            hbo, hbr = nirs_chunk
            if hbo is not None and len(hbo):
                self.ne.push(hbo, hbr)

    # ============================================================ features
    def _S(self, k: np.ndarray) -> np.ndarray:
        """Ковариации окон сетки k, усреднённые по cov_avg_n последним окнам (k−n+1 … k), только прошлое."""
        n = int(self.cfg.eeg.get("cov_avg_n", 1))
        if n <= 1:
            return self.fe.S[k]
        k = np.asarray(k)
        acc = self.fe.S[k].copy()
        for j in range(1, n):
            acc += self.fe.S[np.maximum(k - j, 0)]
        return acc / n

    def _A(self, k: np.ndarray) -> np.ndarray:
        """Вспомогательные (CSP) ковариации окон сетки k, при csp.avg_n > 1 — среднее n последних окон."""
        n = int(self.csp_cfg.get("avg_n", 1))
        k = np.asarray(k)
        acc = self.fe.A[k].copy()
        for j in range(1, n):
            acc += self.fe.A[np.maximum(k - j, 0)]
        return acc / n

    def _csp_block(self, A: np.ndarray) -> np.ndarray:
        """CSP log-var + лог-мощность + асимметрия для батча вспомогательных ковариаций (N, 21, 21)."""
        c = self.csp_cfg
        sym = [tuple(p) for p in c.get("symmetry", [])] if c.get("use_asym", True) else None
        return features_from_cov(transform(A, self.M_csp), self.csp.filters, self.csp.channels, sym,
                                 use_csp=c.get("use_csp", True), use_power=c.get("use_power", True))

    def _S_current(self) -> tuple[np.ndarray, np.ndarray | None]:
        k = self.fe.index_of(self.t)
        S, H = self.fe.current()
        if k is not None and int(self.cfg.eeg.get("cov_avg_n", 1)) > 1:
            S = self._S(np.array([k]))[0]
        return S, H

    def _cov(self, S: np.ndarray, M: np.ndarray, kind: str | None = None, eps: float | None = None) -> np.ndarray:
        e = self.cfg.eeg
        return estimate_cov(transform(S, M), self.win, kind or e["cov"], e["cov_eps"] if eps is None else eps)

    def _ref_mean(self, M: np.ndarray, kind: str, eps: float, mean: str) -> np.ndarray:
        """Среднее ковариаций всех прошлых окон сессии (центр перецентрирования).

        Лог-евклидово среднее считается инкрементально: сумма logm(C_k) по окнам кэшируется для данной
        матрицы M, и в каждом fit_block логарифмируются только новые окна.
        """
        if mean != "logeuclid":
            Call = self._cov(self._S(np.arange(self.fe.K)), M, kind, eps)
            return mean_spd(Call, mean, self.cfg.eeg["riemann_max_n"])
        key = (M.tobytes(), kind, eps)
        c = self._lcache.get(key)
        if c is None:
            c = self._lcache[key] = {"n": 0, "sum": 0.0}
        if self.fe.K > c["n"]:
            new = self._cov(self._S(np.arange(c["n"], self.fe.K)), M, kind, eps)
            c["sum"] = c["sum"] + logm_spd(new).sum(0)
            c["n"] = self.fe.K
        return expm_sym(c["sum"] / c["n"])

    def _session_inputs(self, S: np.ndarray, H: np.ndarray | None,
                        A: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Входы сессионной модели для батча окон S (N, B, 21, 21); A — CSP-ковариации (N, 21, 21)."""
        f = self.feat
        out: dict[str, np.ndarray] = {}
        if f == "logpower":
            d = np.diagonal(transform(S, self.M), axis1=-2, axis2=-1)
            out["X"] = np.log(d + 1e-12).reshape(len(S), -1)
        elif f == "heog":
            out["X"] = np.asarray(H).reshape(len(S), -1)
        elif f in ("ts", "mdm", "ts+mdm"):
            Cw = self.cref_isqrt @ self._cov(S, self.M) @ self.cref_isqrt
            le_mdm = self.cfg.eeg.get("mdm_metric", "riemann") == "logeuclid"
            if f in ("ts", "ts+mdm") or le_mdm:
                out["X"] = ts_vector(Cw)
            if f in ("mdm", "ts+mdm"):
                if le_mdm:
                    out["Xm"] = out["X"]
                    if f == "mdm":
                        del out["X"]
                else:
                    out["Cw"] = Cw
        elif f == "csponly":
            pass  # только блок CSP/мощность/асимметрия (ниже) — векторизатор коллеги как самостоятельный вход
        elif f == "csp":
            C = self._cov(S, self.M)
            out["C"] = C
            if self.csp_W is not None:
                out["X"] = csp_features(C, self.csp_W)
        else:
            raise ValueError(f"eeg.features={f!r}")
        if self.csp is not None and A is not None and self.M_csp is not None:
            extra = self._csp_block(A)          # только во вход sLDA; MDM остаётся на чистой TS-геометрии
            out["X"] = np.hstack([out["X"], extra]) if "X" in out else extra
        return out

    @staticmethod
    def _session_logp(inp: dict[str, np.ndarray], m_lin, m_mdm) -> np.ndarray | None:
        parts = []
        if "X" in inp and m_lin is not None:
            parts.append(m_lin.logp(inp["X"]))
        if "Cw" in inp and m_mdm is not None:
            parts.append(m_mdm.logp(inp["Cw"]))
        if "Xm" in inp and m_mdm is not None:
            parts.append(m_mdm.logp(inp["Xm"]))
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else log_softmax(np.mean(parts, axis=0))

    def _global_logp(self, S: np.ndarray) -> np.ndarray | None:
        if self.glob is None or self.glob_isqrt is None:
            return None
        g = self.glob.meta["eeg"]
        Cw = self.glob_isqrt @ self._cov(S, self.M_glob, g["cov"], g["cov_eps"]) @ self.glob_isqrt
        return self.glob.model.logp(ts_vector(Cw))

    def _mix(self, ls: np.ndarray | None, lg: np.ndarray | None, n_blocks: int) -> np.ndarray:
        if lg is None:
            return ls if ls is not None else UNIFORM[None, :]
        if ls is None:
            return lg
        k = self.cfg.glob["k"]
        w = 1.0 if k == "inf" else float(k) / (float(k) + n_blocks)
        return log_softmax((1.0 - w) * ls + w * lg)

    # ============================================================ fit_block
    def fit_block(self, block_idx: int, labels: dict[str, Any]) -> None:
        if self.ne is not None:
            self.ne.mark(self.t)
        self._add_labels(labels, block_idx)
        self.n_blocks += 1
        if not self.ev_k:
            return
        k = np.asarray(self.ev_k)
        y = np.asarray(self.ev_y)
        tr = np.asarray(self.ev_tr)
        S = self._S(k)
        H = self.fe.H[k] if self.fe.heog_on else None
        e = self.cfg.eeg
        A = None
        if self.csp is not None:
            if self.M_csp is None:  # фиксированный порядок каналов CSP; плохие каналы — интерполяция соседями
                self.M_csp, _ = spatial_matrix(self.fe.names, self.csp.channels, "none",
                                               self.fe.bad_channels(), interpolate=True)
            A = self._A(k)
        # --- маска плохих каналов и референс (по прошлым данным)
        bad = self.fe.bad_channels() if self.feat != "heog" else set()
        self.M, self.ch_sess = spatial_matrix(self.fe.names, self.fe.channels, e["reference"], bad)
        if self.feat in ("ts", "mdm", "ts+mdm"):
            self.cref_isqrt = invsqrtm_spd(self._ref_mean(self.M, e["cov"], e["cov_eps"], e["mean"]))
        if self.feat == "csp":
            self.csp_W = fit_csp(self._cov(S[tr], self.M), y[tr], int(e["csp_n"]))
        inp = self._session_inputs(S, H, A)
        self.m_lin, self.m_mdm = self._fit_session(inp, y, tr)
        # --- глобальная модель: M (с интерполяцией плохих, фиксируется по блоку 1) и Cref:
        #     recenter="block1" — по окнам блока 1 (как при обучении), "all" — по всем прошлым окнам
        recenter_all = self.cfg.glob.get("recenter", "block1") == "all"
        if self.glob is not None and (self.glob_isqrt is None or recenter_all):
            g = self.glob.meta["eeg"]
            if self.M_glob is None:
                self.M_glob, _ = spatial_matrix(self.fe.names, self.fe.channels, g["reference"], bad,
                                                interpolate=True)
            self.glob_isqrt = invsqrtm_spd(self._ref_mean(self.M_glob, g["cov"], g["cov_eps"], g["mean"]))
        # --- NIRS
        Xn = None
        if self.ne is not None and self.ne.n_samples > 0:
            self.ne.refit_linear()
            Xn = self.ne.features(self.fe.ends[k])
            self.m_nirs = fit_linear(Xn[tr], y[tr], self.cfg.nirs["classifier"], self.cfg.nirs["C"],
                                     self.cfg.raw["priors"], self.cfg.raw["random_state"],
                                     self.cfg.nirs.get("logit_T"))
        # --- смещения
        if self.cfg.bias["mode"] == "session" and len(np.unique(self.ev_b)) >= self.cfg.bias["min_blocks"]:
            self._calibrate_session(inp, Xn, y, tr, S)
        self._update_bias()
        self._flush_dump()

    def _fit_session(self, inp: dict[str, np.ndarray], y: np.ndarray, tr: np.ndarray):
        e = self.cfg.eeg
        m_lin = m_mdm = None
        if "X" in inp:
            m_lin = fit_linear(inp["X"][tr], y[tr], e["classifier"], e["C"], self.cfg.raw["priors"],
                               self.cfg.raw["random_state"], e.get("logit_T"))
        if "Cw" in inp:
            m_mdm = fit_mdm(inp["Cw"][tr], y[tr], e["mean"], e["mdm_T"], e["riemann_max_n"])
        if "Xm" in inp:
            m_mdm = fit_ts_mdm(inp["Xm"][tr], y[tr], e["mdm_T"])
        return m_lin, m_mdm

    def _add_labels(self, labels: dict[str, Any], block: int) -> None:
        st = np.asarray(labels["states"])
        idx = np.asarray(labels["eeg_indices"])
        if len(idx) == 0:
            return
        skip = int(round(self.cfg.eeg["train_skip_s"] * self.fs))
        brk = np.flatnonzero((np.diff(idx) != 1) | (np.diff(st) != 0)) + 1
        for seg_i, seg_s in zip(np.split(idx, brk), np.split(st, brk)):
            c = int(seg_s[0])
            if c not in (1, 2, 3):
                continue
            a, b = int(seg_i[0]), int(seg_i[-1]) + 1
            k0 = -(-a // self.step)                       # e − win ≥ a
            k1 = (b - self.win) // self.step              # e ≤ b
            for kk in range(k0, k1 + 1):
                if kk >= self.fe.K or self.fe.ends[kk] != self.win + kk * self.step:
                    continue
                self.ev_k.append(kk)
                self.ev_y.append(c)
                self.ev_b.append(int(block))
                self.ev_tr.append(bool(self.fe.ends[kk] - self.win >= a + skip))

    # ------------------------------------------------------- bias (LOBO)
    def _calibrate_session(self, inp: dict[str, np.ndarray], Xn: np.ndarray | None, y: np.ndarray,
                           tr: np.ndarray, S: np.ndarray) -> None:
        """Leave-one-block-out по прошедшим блокам → вне-выборочные лог-вероятности → смещения."""
        if self.feat == "csp":
            return
        blk = np.asarray(self.ev_b)
        ub = np.unique(blk)
        lg_all = self._global_logp(S)
        le_o = np.zeros((len(y), 3))
        ln_o = np.zeros((len(y), 3)) if Xn is not None else None
        ok = np.ones(len(y), dtype=bool)
        sp = self.cfg.smoothing["space"]
        for bk in ub:
            te = blk == bk
            trk = tr & ~te
            m_lin, m_mdm = self._fit_session(inp, y, trk)
            ls = self._session_logp({key: v[te] for key, v in inp.items()}, m_lin, m_mdm)
            lg = lg_all[te] if lg_all is not None else None
            if ls is None and lg is None:
                ok[te] = False
                continue
            le_o[te] = ema_sequence(self._mix(ls, lg, len(ub) - 1), self.ema_a, sp)
            if ln_o is not None:
                mn = fit_linear(Xn[trk], y[trk], self.cfg.nirs["classifier"], self.cfg.nirs["C"],
                                self.cfg.raw["priors"], self.cfg.raw["random_state"], self.cfg.nirs.get("logit_T"))
                if mn is None:
                    ok[te] = False
                    continue
                ln_o[te] = ema_sequence(mn.logp(Xn[te]), self.ema_an, sp)
        if ok.sum() < 30:
            return
        bc = self.cfg.bias
        grid = tuple(bc["grid"])
        self.b_sess["eeg"] = calibrate_bias(le_o[ok], y[ok], None, grid, bc["passes"])
        if ln_o is not None:
            lh = fuse(le_o[ok], ln_o[ok], self.cfg.fusion["mode"], self._w_eeg(), self.fusion_file.get("stack"))
            self.b_sess["hyb"] = calibrate_bias(lh, y[ok], None, grid, bc["passes"])
            self.b_sess["nirs"] = calibrate_bias(ln_o[ok], y[ok], None, grid, bc["passes"])
        else:
            self.b_sess["hyb"] = self.b_sess["eeg"]

    def _w_eeg(self) -> float:
        return float(self.fusion_file.get("w_eeg", self.cfg.fusion["w_eeg"]))

    def _update_bias(self) -> None:
        bc = self.cfg.bias
        fixed = np.asarray(bc["fixed"], dtype=float)
        for key in ("hyb", "eeg", "nirs"):
            bg = np.zeros(3)
            if bc["mode"] in ("global", "session"):
                bg = np.asarray(self.fusion_file.get("bias", {}).get(key, [0.0, 0.0, 0.0]), dtype=float)
            b = bg
            if bc["mode"] == "session" and key in self.b_sess:
                lam = self.n_blocks / (self.n_blocks + float(bc["shrink_k"]))
                b = lam * self.b_sess[key] + (1.0 - lam) * bg
            self.bias_cur[key] = b + fixed

    # ============================================================= predict
    def predict(self) -> dict[str, Any]:
        S, H = self._S_current()
        S1 = S[None]
        ls = None
        if self.m_lin is not None or self.m_mdm is not None:
            A1 = None
            if self.csp is not None:
                k_cur = self.fe.index_of(self.t)
                A1 = self._A(np.array([k_cur])) if k_cur is not None else self.fe.current_aux()[None]
            ls = self._session_logp(self._session_inputs(S1, None if H is None else H[None], A1),
                                    self.m_lin, self.m_mdm)
        lg = self._global_logp(S1)
        le = self._mix(ls, lg, self.n_blocks)[0]
        le_s = self.ema_eeg(le)
        ln_s = None
        if self.ne is not None:
            self.ne.mark(self.t)
        if self.ne is not None and self.m_nirs is not None:
            ln = self.m_nirs.logp(self.ne.features(np.array([self.t])))[0]
            ln_s = self.ema_nirs(ln)
        lh = fuse(le_s, ln_s, self.cfg.fusion["mode"], self._w_eeg(), self.fusion_file.get("stack"))
        bh = self.bias_cur["hyb"] if ln_s is not None else self.bias_cur["eeg"]
        y = int(CLASSES[np.argmax(lh + bh)])
        y_eeg = int(CLASSES[np.argmax(le_s + self.bias_cur["eeg"])])
        y_nirs = None
        if self.nirs_on:
            y_nirs = int(CLASSES[np.argmax(ln_s + self.bias_cur["nirs"])]) if ln_s is not None else 1
        if self.dump_dir:
            nan3 = [np.nan] * 3
            self.dump_rows.append([self.t] + (list(ls[0]) if ls is not None else nan3)
                                  + (list(lg[0]) if lg is not None else nan3) + list(le_s)
                                  + (list(ln_s) if ln_s is not None else nan3) + list(lh)
                                  + [y, y_eeg, y_nirs if y_nirs is not None else -1])
        return {"y": y, "y_eeg": y_eeg, "y_nirs": y_nirs, "p": tuple(float(v) for v in np.exp(lh))}

    # ================================================================ dump
    def _flush_dump(self) -> None:
        if not self.dump_dir or not self.dump_rows:
            return
        d = Path(self.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / (Path(str(self.meta.get("session_name", "session"))).stem + ".csv")
        head = ("end," + ",".join(f"{n}{i}" for n in ("ls", "lg", "le", "ln", "lh") for i in (1, 2, 3))
                + ",y,y_eeg,y_nirs")
        with p.open("a" if self.dump_started else "w", encoding="utf-8") as fh:
            if not self.dump_started:
                fh.write(head + "\n")
            for r in self.dump_rows:
                fh.write(",".join("nan" if (isinstance(v, float) and np.isnan(v)) else
                                  (str(int(v)) if i == 0 or i >= 16 else f"{v:.5f}") for i, v in enumerate(r)) + "\n")
        self.dump_started = True
        self.dump_rows = []
