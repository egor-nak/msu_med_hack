"""Инвариантность к разбиению push (защита от утечки через тайминг, ТЗ §3.4 / L3–L4).

Повторяет порядок событий run.py, но (A) дробит каждую порцию push на куски ровно по 62 отсчёта
(последний — остаток) и (B) на подотрезке из 2000 отсчётов подаёт данные по 1 отсчёту. В каждой
точке, где в оригинале вызывается predict, вызывается predict. Ответы y и y_nirs должны совпасть
с оригинальным прогоном на 100 %.

  python tests/test_chunk_invariance.py [session.mat] [--legacy mvp_classifier]
"""
from __future__ import annotations

import sys

import numpy as np

from sim import default_session, iu, simulate


def pieces62(cur: int, end: int) -> list[int]:
    return list(range(cur + 62, end, 62)) + [end]


def make_single(lo: int, hi: int):
    def cut(cur: int, end: int) -> list[int]:
        a, b = max(cur, lo), min(end, hi)
        if a >= b:
            return [end]
        return [a] + list(range(a + 1, b + 1)) + [end]
    return cut


def compare(a, b) -> tuple[int, int, int]:
    assert [e for e, _ in a] == [e for e, _ in b], "разный набор окон"
    dy = sum(x["y"] != y["y"] for (_, x), (_, y) in zip(a, b))
    dn = sum(x.get("y_nirs") != y.get("y_nirs") for (_, x), (_, y) in zip(a, b))
    return len(a), dy, dn


def run(path, cls=None) -> dict:
    s = iu.load_session(path)
    base = simulate(s, cls=cls)
    n, dy, dn = compare(base, simulate(s, pieces62, cls=cls))
    ends = [e for e, _ in base]
    lo = ends[len(ends) // 2] - 1000
    n2, dy2, dn2 = compare(base, simulate(s, make_single(lo, lo + 2000), cls=cls))
    return {"session": s["name"], "n": n, "diff_y_62": dy, "diff_nirs_62": dn, "diff_y_1": dy2, "diff_nirs_1": dn2,
            "ok": dy == dn == dy2 == dn2 == 0}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cls = None
    if "--legacy" in sys.argv:
        import importlib
        cls = importlib.import_module("legacy." + sys.argv[sys.argv.index("--legacy") + 1]).OnlineClassifier
    f = args[0] if args else default_session()
    r = run(f, cls)
    print(r)
    assert r["ok"], "ОТВЕТЫ ЗАВИСЯТ ОТ РАЗБИЕНИЯ PUSH (утечка тайминга)"
    print("OK: ответы инвариантны к разбиению push")
