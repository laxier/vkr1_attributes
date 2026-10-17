"""Контрольные варианты без модели: «всегда null» и «самое частое значение по атрибуту».

Запуск (из папки со скриптом):
    python baselines.py pilot
    python baselines.py real_val
Самое частое значение берётся по эталону пилота (val и test для этого не читаются).
Эталон берётся из любого файла результатов набора: он одинаков у всех запусков.
Метрики те же, что в ноутбуке: micro-F1, macro-F1, exact, ложные на null.
Результат дописывается в results/baselines.csv папки этапа (строка набора заменяется).
"""
import glob, json, os, sys, time
from collections import Counter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ATTRS = ["диагональ", "разрешение", "частота", "тип матрицы", "яркость", "контрастность", "изогнутость"]
STAGE = {"pilot": "06_pilot", "real_val": "07_main_run", "heldout": "07_main_run", "real_test": "09_final"}


def gold(split):
    """{id карточки: эталон} из первого попавшегося файла результатов набора"""
    files = sorted(glob.glob(os.path.join(HERE, "..", STAGE[split], "results", f"{split}__*.jsonl")))
    if not files:
        sys.exit(f"Набор {split} ещё не прогоняли: нет файлов {split}__*.jsonl.")
    return {r["id"]: r["gold"] for r in map(json.loads, open(files[0], encoding="utf-8"))}


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def score(golds, answer):
    """answer: эталон -> ответ варианта; метрики как в ноутбуке (нормализация не нужна: ответы уже в виде таблицы)"""
    cnt = {a: [0, 0, 0] for a in ATTRS}
    false_null, exact = [], []
    for g in golds.values():
        p, ok = answer(g), True
        for a in ATTRS:
            if g[a] is None:
                false_null.append(p[a] is not None)
                if p[a] is not None:
                    cnt[a][1] += 1; ok = False
            elif p[a] == g[a]:
                cnt[a][0] += 1
            else:
                cnt[a][2] += 1; ok = False
                if p[a] is not None:
                    cnt[a][1] += 1
        exact.append(ok)
    tp, fp, fn = (sum(c[i] for c in cnt.values()) for i in range(3))
    return {"micro_f1": round(f1(tp, fp, fn), 3), "macro_f1": round(np.mean([f1(*c) for c in cnt.values() if sum(c)]), 3),
            "exact": round(np.mean(exact), 3), "ложные_на_null": round(np.mean(false_null), 3)}


def main():
    split = sys.argv[1]
    pilot = gold("pilot")
    top = {a: Counter(g[a] for g in pilot.values() if g[a] is not None).most_common(1)[0][0] for a in ATTRS}
    golds = gold(split)
    rows = [{"split": split, "вариант": "всегда null", "n": len(golds), **score(golds, lambda g: {a: None for a in ATTRS})},
            {"split": split, "вариант": "самое частое (по пилоту)", "n": len(golds), **score(golds, lambda g: dict(top))}]
    print("самое частое значение по пилоту:", top)
    print(pd.DataFrame(rows).to_string(index=False))
    path = os.path.join(HERE, "..", STAGE[split], "results", "baselines.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(old):
        old = old[old["split"] != split]
    pd.concat([old, pd.DataFrame(rows)]).to_csv(path, index=False)
    print("записано:", os.path.normpath(path), time.strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
