"""Парный bootstrap по карточкам: интервал для запуска и для разницы двух запусков.

Запуск (из папки 05_protocol):
    python bootstrap.py pilot gemma4_e4b/few-shot/schema
    python bootstrap.py real_val gemma4_e4b/few-shot/schema qwen35_9b/few-shot/schema
    python bootstrap.py real_val A B groups.json     # groups.json: {"id карточки": "группа", ...}, файл можно указать в любом месте

Запуск задаётся как модель/режим/формат (формат: free или schema).
Метрики: micro-F1 с нормализацией и доля ложных значений на null.
4000 повторов, seed 0, интервал 95%. Без файла групп карточки берутся по одной,
с файлом групп пересэмплируются группы почти-дубликатов целиком.
"""
import json, re, sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATTRS = ["диагональ", "разрешение", "частота", "тип матрицы", "яркость", "контрастность", "изогнутость"]
B = 4000

# norm() берём из ноутбука, чтобы метрики совпадали с summary.csv
nb = json.load(open(f"{HERE}/05_run_model.ipynb", encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
exec([c for c in code if "def parse(raw)" in c][0])


def per_card(split, run):
    """по каждой карточке: tp, fp, fn, число ложных значений на null, число null в эталоне"""
    model, mode, fmt = run.split("/")
    rows = {}
    for line in open(f"{HERE}/results/{split}__{model}__{mode}__{fmt}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows.setdefault(r["id"], r)              # дубли по id: первая запись
    ids = sorted(rows)
    out = []
    for i in ids:
        pred = rows[i]["pred"] or {a: None for a in ATTRS}
        tp = fp = fn = fnull = nnull = 0
        for a in ATTRS:
            g, p = rows[i]["gold"][a], pred[a]
            if p is not None:
                p = norm(a, p)
            if g is None:
                nnull += 1
                if p is not None:
                    fp += 1; fnull += 1
            elif p == g:
                tp += 1
            else:
                fn += 1
                fp += p is not None
        out.append([tp, fp, fn, fnull, nnull])
    return ids, np.array(out, float)


def metrics(w, x):
    """w: веса карточек (повторы x карточек), x: счётчики -> micro-F1 и ложные на null"""
    tp, fp, fn, fnull, nnull = (w @ x).T
    return 2 * tp / (2 * tp + fp + fn), fnull / nnull


def show(name, point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:22s} {point:+.3f}  [{lo:+.3f}; {hi:+.3f}]" + ("  (интервал содержит 0)" if lo <= 0 <= hi else ""))


split = sys.argv[1]
runs = [a for a in sys.argv[2:] if "/" in a and not a.endswith(".json")]
gf = [a for a in sys.argv[2:] if a.endswith(".json")]
groups = json.load(open(gf[0], encoding="utf-8")) if gf else None

data = [per_card(split, r) for r in runs]
assert len(runs) == 1 or data[0][0] == data[1][0], "наборы карточек разные"
ids = data[0][0]
gid = np.array([groups[i] if groups else str(k) for k, i in enumerate(ids)])
_, gid = np.unique(gid, return_inverse=True)     # номера групп 0..G-1
G = gid.max() + 1

rng = np.random.default_rng(0)
draw = rng.integers(0, G, (B, G))                # какие группы попали в выборку
w = np.stack([np.bincount(d, minlength=G) for d in draw])[:, gid]     # вес каждой карточки в каждом повторе
one = np.ones((1, len(ids)))

print(f"{split}: {len(ids)} карточек, {G} групп, {B} повторов")
res = []
for r, (_, x) in zip(runs, data):
    f, fn_ = metrics(w, x)
    f0, fn0 = (m[0] for m in metrics(one, x))
    res.append((f, fn_, f0, fn0))
    print(r)
    show("micro-F1", f0, f); show("ложные на null", fn0, fn_)
if len(runs) == 2:
    print(f"разница: {runs[0]} минус {runs[1]}")
    show("micro-F1", res[0][2] - res[1][2], res[0][0] - res[1][0])
    show("ложные на null", res[0][3] - res[1][3], res[0][1] - res[1][1])
