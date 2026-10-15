"""Генерация синтетических описаний мониторов через Claude.

Запуск (из папки 04_synthetic/):
    ключ лежит в ../.env (строка ANTHROPIC_API_KEY=...), файл в .gitignore
    python generate.py fewshot 20 1      # имя набора, сколько штук, seed
    python generate.py heldout 80 2
    python generate.py train 2000 3

Результат: ../data/synthetic/<имя>.jsonl, строка = {"description", "gold", "формат"}.
Эталон известен по построению, потом проверяется теми же правилами, что и на реальных карточках.
Из реальных данных берётся только пилот (100 карточек): стиль, частота упоминаний и значения.
"""
import json, os, random, re, sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv("../.env")   # кладёт ANTHROPIC_API_KEY в окружение, anthropic его подхватит сам

MODEL = "sonnet"                                  # какой моделью генерировать: "sonnet" или "haiku"

EDA = "../03_eda/outputs/dns_cards_500"          # gold_pairs.csv и split_ids.json
CARDS = "../data/dns_cards_500.json"
STARTS = ["с названия бренда и модели", "с назначения монитора (игры, работа, дизайн)", "с самой заметной особенности",
          "с короткой фразы-призыва", "с описания внешнего вида и корпуса"]   # чтобы описания начинались по-разному
N_STYLE = 3                                       # сколько реальных описаний показать в промпте
ATTRS = ["диагональ", "разрешение", "частота", "тип матрицы", "яркость", "контрастность", "изогнутость"]

# --- проверка эталона: те же правила, что в 03_eda.ipynb ---
def number(s):
    return re.search(r"\d+(?:[.,]\d+)?", s).group().replace(",", ".")

def resolution(s):
    w, h = re.findall(r"\d+", s)[:2]
    return f"{w}x{h}" if int(w) >= 640 else None

def contrast(s):
    s = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", s.lower())
    v = float(number(s)) * (10**6 if re.search(r"m|м", s) else 1)
    return str(int(v)) if v >= 100 else None

NORM = {"диагональ": lambda s: f"{float(number(s)):g}", "разрешение": resolution, "частота": number,
        "тип матрицы": lambda s: s.upper().replace("QD-", ""), "яркость": number, "контрастность": contrast}
PATTERN = {"диагональ": r'(\d+(?:[.,]\d+)?)\s*-?\s*(?:"|”|″|дюйм)',
           "разрешение": r"(\d{3,4}\s*[xх×*]\s*\d{3,4})",
           "частота": r"(\d{2,3})\s*(?:гц|герц)",
           "тип матрицы": r"\b(ips|va|(?:qd-)?oled|tn)\b",
           "яркость": r"(\d{3,4})\s*(?:кд|нит|cd)",
           "контрастность": r"((?:\d{1,3}(?:\s\d{3})+|\d+(?:[.,]\d+)?)\s*(?:m|млн)?)\s*(?::|к|/)\s*1\b",
           "изогнутость": r"(изогнут|curved|плоск)"}
CURVED = {"изогнут": "да", "curved": "да", "плоск": "нет"}

def found_in_text(attr, text):
    text = text.lower().replace("ё", "е")
    found = set()
    for m in re.findall(PATTERN[attr], text):
        v = CURVED[m] if attr == "изогнутость" else NORM[attr](m)
        if v: found.add(v)
    return found

def check(text, gold):
    """пустая строка, если всё в порядке; иначе что не так"""
    for attr in ATTRS:
        found = found_in_text(attr, text)
        want = {gold[attr]} if gold[attr] else set()
        if found != want:
            return f"{attr}: нужно {sorted(want) or 'не называть'}, в тексте {sorted(found) or 'нет'}"
    return ""

# --- что просим написать ---
def make_gold(rng, pool, p):
    gold = {}
    for attr in ATTRS:
        gold[attr] = rng.choice(pool[attr]) if rng.random() < p[attr] else None
    return gold

def make_format(rng, attr, v):
    """как именно записать значение в тексте (форматы взяты из анализа 03)"""
    if attr == "диагональ":
        s = v.replace(".", ",") if "." in v and rng.random() < 0.15 else v
        return rng.choice([f'{s}"', f"{s}-дюймовый", f"{s} дюймов"])
    if attr == "разрешение":
        w, h = v.split("x")
        return f"{w}{rng.choice(['x', 'x', 'x', 'х', '×'])}{h}"
    if attr == "частота":
        return rng.choice([f"{v} Гц", f"{v} Гц", f"{v}Гц"])
    if attr == "яркость":
        return rng.choice([f"{v} кд/м²", f"{v} нит", f"{v} cd/m2"])
    if attr == "контрастность":
        return f"{v}:1"
    if attr == "изогнутость":
        return "изогнутый" if v == "да" else "плоский"
    return v   # тип матрицы пишем как есть

def make_prompt(gold, fmt, style, brand, start, length):
    must = "\n".join(f"- {a}: {fmt[a]}" for a in ATTRS if gold[a])
    never = ", ".join(a for a in ATTRS if not gold[a]) or "нет"
    examples = "\n\n".join(f"Пример {i + 1}:\n{t}" for i, t in enumerate(style))
    return f"""Напиши описание монитора для карточки товара интернет-магазина электроники, по-русски.
Бренд: {brand}. Название модели придумай сам (в стиле артикулов, например «27G2» или «XG273QZ»).
Начни описание {start}.
Объём около {length} символов (не больше чем на 15% длиннее), связный текст в стиле примеров ниже: маркетинговые фразы, порты, подсветка, эргономика, режимы и т.п.

В тексте обязательно назови эти характеристики, именно в такой записи:
{must}

Эти характеристики не называй совсем (ни числом, ни словом, ни намёком): {never}.
Каждую обязательную характеристику называй с одним и тем же значением, без противоречий.

Реальные описания для стиля (не копируй из них фразы):

{examples}

Ответь только текстом описания, без заголовка и пояснений."""

# размышление выключаем, чтобы условия были одинаковыми и не оплачивались лишние токены.
# Sonnet 5.5 не принимает "disabled", поэтому у моделей разные функции.
def ask_sonnet(client, prompt):
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=2000, thinking={"type": "between_tools"},
                               messages=[{"role": "user", "content": prompt}])
    print("токены: вход", r.usage.input_tokens, "выход", r.usage.output_tokens)   # для оценки затрат
    return "".join(b.text for b in r.content if b.type == "text").strip()

def ask_haiku(client, prompt):
    r = client.messages.create(model="claude-haiku-5-5", max_tokens=2000, thinking={"type": "disabled"},
                               messages=[{"role": "user", "content": prompt}])
    print("токены: вход", r.usage.input_tokens, "выход", r.usage.output_tokens)
    return "".join(b.text for b in r.content if b.type == "text").strip()

ask = ask_sonnet if MODEL == "sonnet" else ask_haiku

def grams(text, n=6):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

def main(name, n, seed):
    import anthropic
    client = anthropic.Anthropic()
    rng = random.Random(seed)

    # только пилот: частоты и значения из gold_pairs, стиль из текстов пилота
    pilot = set(json.load(open(f"{EDA}/split_ids.json"))["pilot"])
    pairs = pd.read_csv(f"{EDA}/gold_pairs.csv")
    pairs = pairs[pairs["url"].isin(pilot)]
    p = {a: pairs[pairs["атрибут"] == a]["эталон"].notna().mean() for a in ATTRS}
    pool = {a: pairs[pairs["атрибут"] == a]["эталон"].dropna().astype(str).tolist() for a in ATTRS}
    cards = [c for c in json.load(open(CARDS, encoding="utf-8")) if c["url"] in pilot]
    real = [c["description"] for c in cards]
    brands = sorted({c["category"].split(" / ")[-1] for c in cards})

    os.makedirs("../data/synthetic", exist_ok=True)
    path = f"../data/synthetic/{name}.jsonl"
    done = sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0
    bad = 0
    while done < n:
        gold = make_gold(rng, pool, p)
        fmt = {a: make_format(rng, a, v) for a, v in gold.items() if v}
        style = rng.sample(real, N_STYLE)
        length = min(max(50 * round(len(rng.choice(real)) / 50), 250), 1300)   # длина как у случайного реального описания
        text = ask(client, make_prompt(gold, fmt, style, rng.choice(brands), rng.choice(STARTS), length))
        reason = check(text, gold) or ("скопирована фраза из образца" if any(grams(text) & grams(s) for s in style) else "")
        if reason:
            bad += 1
            print("отбраковано:", bad, "| принято:", done, "|", reason)
            continue
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"description": text, "gold": gold, "формат": fmt}, ensure_ascii=False) + "\n")
        done += 1
        print(done, "/", n)
    print("готово:", path, "| отбраковано:", bad)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
