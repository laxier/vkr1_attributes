"""Парный bootstrap по карточкам: интервалы для запусков и для разницы двух запусков.

Запуск (из папки 05_protocol), запуск задаётся как модель/режим/формат (формат: free или schema):
    python bootstrap.py pilot gemma4_e4b/few-shot/schema                    # интервал одного запуска
    python bootstrap.py real_val gemma4_e4b/few-shot/schema qwen35_9b/few-shot/schema   # разница двух
    python bootstrap.py real_val all                                        # H2 и H3 для всех моделей
Последним аргументом можно дать groups.json {"id карточки": "группа"}: тогда пересэмплируются
группы почти-дубликатов целиком, а не отдельные карточки.

Метрики: micro-F1 с нормализацией и доля ложных значений на null.
4000 повторов, seed 0, интервал 95%. Разница считается на одних и тех же карточках.
Каждый удавшийся запуск дописывается в results/bootstrap_log.txt (дата, команда, результат).
"""
import json, re, sys, os, time, io, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATTRS = ["диагональ", "разрешение", "частота", "тип матрицы", "яркость", "контрастность", "изогнутость"]
MODELS = ["qwen35_9b", "gemma4_e4b", "llama31_8b", "ministral3_8b", "phi4_mini", "gigachat31_10b", "tlite21_8b", "yandex5_lite_8b"]
B = 4000

# norm() берём из ноутбука, чтобы метрики совпадали с summary.csv
nb = json.load(open(f"{HERE}/05_run_model.ipynb", encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
exec([c for c in code if "def parse(raw)" in c][0])


def per_card(split, run):
    """по каждой карточке: tp, fp, fn, число ложных значений на null, число null в эталоне"""
    model, mode, fmt = run.split("/")
    path = f"{HERE}/results/{split}__{model}__{mode}__{fmt}.jsonl"
    if not os.path.exists(path):
        sys.exit(f"Нет результатов для {run} на наборе {split}: файла {os.path.basename(path)} нет в results/. Этот набор ещё не прогоняли.")
    rows = {}
    for line in open(path, encoding="utf-8"):
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
    """w: веса карточек (повторы x карточки), x: счётчики -> micro-F1 и ложные на null"""
    tp, fp, fn, fnull, nnull = (w @ x).T
    return 2 * tp / (2 * tp + fp + fn), fnull / nnull


def show(name, point, boot, note=""):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:16s} {point:+.3f}  [{lo:+.3f}; {hi:+.3f}]" + ("  (интервал содержит 0)" if lo <= 0 <= hi else "") + note)


def compare(split, runs, groups, quiet=False):
    """runs: один запуск или два; печатает интервалы, возвращает (разница F1, интервал) для двух"""
    data = [per_card(split, r) for r in runs]
    assert len(runs) == 1 or data[0][0] == data[1][0], "наборы карточек разные"
    ids = data[0][0]
    gid = np.array([groups[i] if groups else str(k) for k, i in enumerate(ids)])
    _, gid = np.unique(gid, return_inverse=True)         # номера групп 0..G-1
    G = gid.max() + 1
    draw = np.random.default_rng(0).integers(0, G, (B, G))           # какие группы попали в выборку
    w = np.stack([np.bincount(d, minlength=G) for d in draw])[:, gid]  # вес карточки в каждом повторе
    one = np.ones((1, len(ids)))
    res = []
    if not quiet:
        print(f"{split}: {len(ids)} карточек, {G} групп, {B} повторов")
    for r, (_, x) in zip(runs, data):
        f, fn_ = metrics(w, x)
        f0, fn0 = (m[0] for m in metrics(one, x))
        res.append((f, fn_, f0, fn0))
        if not quiet:
            print(r); show("micro-F1", f0, f); show("ложные на null", fn0, fn_)
    if len(runs) == 2:
        d_f, d_fn = res[0][0] - res[1][0], res[0][1] - res[1][1]
        p_f, p_fn = res[0][2] - res[1][2], res[0][3] - res[1][3]
        if not quiet:
            print(f"разница: {runs[0]} минус {runs[1]}")
            show("micro-F1", p_f, d_f); show("ложные на null", p_fn, d_fn)
        return p_f, d_f


def main():
    split = sys.argv[1]
    args = sys.argv[2:]
    gf = [a for a in args if a.endswith(".json")]
    groups = json.load(open(gf[0], encoding="utf-8")) if gf else None
    runs = [a for a in args if not a.endswith(".json")]
    if runs == ["all"]:
        if not any(f.startswith(split + "__") for f in os.listdir(f"{HERE}/results")):
            sys.exit(f"Набор {split} ещё не прогоняли: в results/ нет файлов {split}__*.jsonl.")
        print(f"{split}: разница micro-F1 (для H2 нужно не меньше +0,02 и интервал без 0; для H3 смотрим, не хуже ли -0,02)")
        for m in MODELS:
            for title, a, b in [("H2 few-shot минус zero-shot, свободно", "few-shot/free", "zero-shot/free"),
                                ("H2 few-shot минус zero-shot, схема", "few-shot/schema", "zero-shot/schema"),
                                ("H3 схема минус свободно, zero-shot", "zero-shot/schema", "zero-shot/free"),
                                ("H3 схема минус свободно, few-shot", "few-shot/schema", "few-shot/free")]:
                A, Bb = f"{m}/{a}", f"{m}/{b}"
                if not all(os.path.exists(f"{HERE}/results/{split}__{r.replace('/', '__')}.jsonl") for r in (A, Bb)):
                    continue
                print(f"{m:16s} {title}")
                p, d = compare(split, [A, Bb], groups, quiet=True)
                show("micro-F1", p, d)
    else:
        compare(split, runs, groups)


class Tee:
    """пишет и на экран, и в буфер"""
    def __init__(self, *files): self.files = files
    def write(self, text): [f.write(text) for f in self.files]
    def flush(self): [f.flush() for f in self.files]


if __name__ == "__main__":
    buf = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buf)):
        main()
    # в журнал попадают только удавшиеся запуски (при ошибке sys.exit пропускает эту запись)
    with open(f"{HERE}/results/bootstrap_log.txt", "a", encoding="utf-8") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M')}  python bootstrap.py {' '.join(sys.argv[1:])}\n{buf.getvalue()}\n")
