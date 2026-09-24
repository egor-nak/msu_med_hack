"""Сборка и исполнение ноутбуков assets/*.ipynb (с сохранёнными выводами).

  python tools/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'assets' else Path.cwd()
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT / 'solution'))
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
EXP = ROOT / 'results' / 'experiments'; FIG = ROOT / 'results' / 'figures'"""


def nb1() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(
        "# 01. EDA и контроли артефактов\n\nВоспроизводит фигуры 1–5 (§7) из кэша EDA (`tools/eda_figures.py`; "
        "при отсутствии кэша пересчитывает по `data/train`) и таблицу контрольных экспериментов §5. "
        "В анализе допустимы двунаправленные фильтры и метки — это не классификатор."))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell(
        "import eda_figures as ED, make_figures as MF\nE = ED.compute()\nMF.eda_figures(E)\n"
        "print('сессий в EDA:', len(E['names']), '; NIRS 12.5 Гц:', len(E['nirs']))"))
    for f, t in [("erd_tfr_c3_c4_cz.png", "1. ERD/ERS на C3/Cz/C4"), ("erd_topomaps.png", "2. Топографии μ/β"),
                 ("fes_artifact_spectrum.png", "3. Артефакт ФЭС ≈ 83 Гц"), ("heog_by_class.png", "4. Горизонтальная ЭОГ"),
                 ("nirs_grand_average.png", "5. NIRS grand average")]:
        c.append(nbf.v4.new_markdown_cell(f"## {t}"))
        c.append(nbf.v4.new_code_cell(f"display(Image(filename=str(FIG / '{f}')))"))
    c.append(nbf.v4.new_markdown_cell("## Числа EDA: сила ФЭС по пробам и ранний ERD"))
    c.append(nbf.v4.new_code_cell(
        "fes = pd.read_csv(ROOT / 'results/cache/fes_segments.csv')\nim = fes[fes.cls > 1]\n"
        "print('медиана прироста 81.5–85.5 Гц в пробах воображения, дБ:', round(im.fes_db.median(), 2))\n"
        "print('доля проб < 2 дБ:', round((im.fes_db < 2).mean(), 3), ' > 6 дБ:', round((im.fes_db > 6).mean(), 3))\n"
        "neuro = ROOT / 'results/cache/neuro_session.csv'\n"
        "if neuro.exists():\n    N = pd.read_csv(neuro)\n"
        "    print('μ-латерализация, медианный AUC(Л vs П) по log C3 − log C4:', round(N.mu_lat_auc.median(), 3))\n"
        "    print('ранний (0.5–2.5 с) μ-ERD относительно покоя, дБ:', round(N.early_erd_db.mean(), 2))"))
    c.append(nbf.v4.new_markdown_cell("## Таблица контролей §5"))
    c.append(nbf.v4.new_code_cell(
        "rows = []\nfor name in ['final', 'final_ch_sm9', 'final_ch_all17', 'final_band40', 'final_band100_LEAKDEMO', 'B3', "
        "'B3_ch_all17', 'B3_band100_LEAKDEMO']:\n"
        "    m = json.load(open(EXP / 'E11_controls' / name / 'metrics_extra.json'))['mean_over_sessions']\n"
        "    rows.append({'вариант': name, 'F': m['hyb_min_f1'], 'P': m['hyb_recall'], 'EEG F': m['eeg_min_f1'],\n"
        "                 'AUC покой/вообр.': m['eeg_auc_rest_im'], 'AUC Л/П': m['eeg_auc_lr'],\n"
        "                 'F 1.0–2.5 с': m['hyb_minf1_t1.0'], 'AUC п/в 1.0–2.5 с': m['eeg_auc_rest_im_t1.0']})\n"
        "for name in ['conf_heog', 'conf_fes', 'conf_frontocc']:\n"
        "    m = json.load(open(EXP / 'E02_conf' / name / 'metrics_extra.json'))['mean_over_sessions']\n"
        "    rows.append({'вариант': name, 'F': m['hyb_min_f1'], 'P': m['hyb_recall'], 'EEG F': m['eeg_min_f1'],\n"
        "                 'AUC покой/вообр.': m['eeg_auc_rest_im'], 'AUC Л/П': m['eeg_auc_lr']})\n"
        "pd.DataFrame(rows).round(3)"))
    c.append(nbf.v4.new_code_cell(
        "for f in ['permutations.json', 'fes_strat_final.json', 'corr_final.json', 'neuro_final.json', 'haufe.json']:\n"
        "    p = EXP / 'E11_controls' / f\n"
        "    if p.exists():\n        d = json.load(open(p))\n"
        "        display(Markdown(f'**{f}**'))\n"
        "        print(json.dumps(d if f != 'haufe.json' else {k: v for k, v in d.items() if k.startswith('L-R')}, "
        "ensure_ascii=False, indent=1)[:2500])"))
    c.append(nbf.v4.new_code_cell(
        "for f in ['haufe_patterns.png', 'controls_scatter.png', 'time_in_segment.png']:\n"
        "    if (FIG / f).exists(): display(Image(filename=str(FIG / f)))"))
    return nb


def nb2() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(
        "# 02. Результаты: сводка экспериментов\n\nВсе числа — онлайн-симуляция замороженных `run.py`+`score.py` "
        "на 56 сессиях train. Сводная таблица — `results/summary.md`, хронология — `research/WORKLOG.md`."))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell(
        "m = json.load(open(ROOT / 'results/metrics/metrics.json'))\nl = json.load(open(EXP / 'E12_final_loso/final/metrics.json'))\n"
        "mvp = json.load(open(EXP / 'E00_mvp/metrics.json'))\n"
        "pd.DataFrame([{'прогон': n, 'F': x['hybrid']['min_f1'], 'P': x['hybrid']['macro_recall'], "
        "'NIRS F': x['nirs_only']['min_f1'], 'EEG F': x['eeg_only_info']['min_f1'], 'штраф': x['timings']['penalty_total'], "
        "'балл': x['score']['total']} for n, x in [('MVP (утечка Δt)', mvp), ('финал, LOSO', l), "
        "('финал, in-sample (results/metrics)', m)]]).round(4)"))
    c.append(nbf.v4.new_code_cell(
        "S = pd.read_csv(ROOT / 'results/summary_all.csv')\nS.sort_values('балл', ascending=False).head(25)"))
    c.append(nbf.v4.new_code_cell(
        "print(json.dumps(json.load(open(EXP / 'E12_final_loso/nested.json'))['nested'], indent=1))\n"
        "cmp = EXP / 'E12_final_loso/compare_mvp.json'\nif cmp.exists(): print(cmp.read_text())"))
    c.append(nbf.v4.new_code_cell(
        "d = pd.read_csv(EXP / 'E12_final_loso/final/per_session_extra.csv')\n"
        "d[['session', 'hyb_min_f1', 'hyb_recall', 'eeg_min_f1', 'nirs_min_f1', 'eeg_auc_rest_im', 'eeg_auc_lr']].round(3)"))
    for f in ["pipelines_comparison.png", "ablation_waterfall.png", "learning_curve_blocks.png", "modalities.png",
              "confusion_examples.png"]:
        c.append(nbf.v4.new_code_cell(f"display(Image(filename=str(FIG / '{f}')))"))
    return nb


def main() -> None:
    A.mkdir(exist_ok=True)
    for name, nb in (("01_eda_and_controls.ipynb", nb1()), ("02_results.ipynb", nb2())):
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(A)}})
        nbf.write(nb, A / name)
        print("→", A / name)


if __name__ == "__main__":
    main()
