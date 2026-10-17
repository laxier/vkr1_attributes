# Длина входного промпта в токенах для пилота, без GPU и без повторного прогона.
# Вставить ячейкой в ноутбук ПОСЛЕ ячеек с данными и промптом (нужны MODELS, get_items, make_prompt, OUT, pd, os, json).
# SPLITS в ноутбуке может быть любым: пилотные карточки ячейка берёт сама. Для Llama и Gemma нужен HF_TOKEN.
# Результат: <OUT>/prompt_tokens_pilot.csv (повторный запуск обновляет только пересчитанные модели); положить в 06_pilot/results/.
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

pilot = get_items("pilot")
rows = []
for key, name in MODELS.items():
    try:
        kw = {"fix_mistral_regex": True} if key == "ministral3_8b" else {}
        if key == "gigachat31_10b":        # конфиг с int вместо float, как в ячейке загрузки; веса не качаем, только токенизатор
            src = snapshot_download(name, allow_patterns=["*.json", "*.model", "*.txt", "*.py", "*.jinja"])
            fixed = "/content/gigachat_tok"
            os.makedirs(fixed, exist_ok=True)
            for f in os.listdir(src):
                if f != "config.json" and not os.path.exists(f"{fixed}/{f}"):
                    os.symlink(os.path.realpath(f"{src}/{f}"), f"{fixed}/{f}")
            cfg = json.load(open(f"{src}/config.json"))
            cfg["routed_scaling_factor"] = float(cfg["routed_scaling_factor"])
            json.dump(cfg, open(f"{fixed}/config.json", "w"), indent=2)
            name = fixed
        tok = AutoTokenizer.from_pretrained(name, **kw)
        for mode in ("zero-shot", "few-shot"):
            n = []
            for it in pilot:
                chat = [{"role": "user", "content": make_prompt(it["text"], mode)}]
                text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                n.append(len(tok(text, add_special_tokens=False)["input_ids"]))      # так же, как в generate
            rows.append({"model": key, "mode": mode, "n": len(n), "токенов_вход_среднее": round(sum(n) / len(n), 1)})
        print(key, "готово")
    except Exception as e:
        print(key, "ОШИБКА:", repr(e))

path = f"{OUT}/prompt_tokens_pilot.csv"
new = pd.DataFrame(rows)
if len(new):
    try:
        old = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):    # файла нет или он пустой
        old = pd.DataFrame()
    if len(old):
        old = old[~old.set_index(["model", "mode"]).index.isin(new.set_index(["model", "mode"]).index)]   # старые строки этих моделей заменяем
    pd.concat([old, new]).to_csv(path, index=False)
print(pd.read_csv(path))
