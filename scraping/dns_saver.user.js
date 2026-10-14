// ==UserScript==
// @name         DNS cards saver
// @namespace    vkr1_attributes
// @version      0.5
// @description  Сохраняет карточки DNS (описание + характеристики) для ВКР. Написан с помощью ИИ.
// @match        https://www.dns-shop.ru/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==
(() => {
  const TARGET = 500;     // сколько карточек нужно
  const MIN_DESC = 200;   // короче описание — карточка не нужна
  const MAX_TRIES = 10;   // столько тиков (по 1.5 с) ждём загрузку карточки
  const PAUSE = [2000, 4000]; // пауза перед переходом к следующей карточке, мс

  const text = el => (el ? el.innerText.trim() : '');
  const load = (k, d) => JSON.parse(localStorage.getItem(k) || d);
  const save = (k, v) => localStorage.setItem(k, JSON.stringify(v));
  const isProduct = () => /\/product\/[0-9a-f]+\//.test(location.pathname);
  const clean = u => u.split(/[?#]/)[0].replace(/(characteristics|opinion)\/?$/, '');

  // описание: сам текст, без кнопки «Развернуть описание»
  function readDescription() {
    return text(document.querySelector('.product-card-description__text-description'));
  }

  // характеристики: { "Параметр": "Значение" }
  function readSpecs() {
    const specs = {};
    document.querySelectorAll('.product-characteristics__spec').forEach(r => {
      const k = text(r.querySelector('.product-characteristics__spec-title'));
      const v = text(r.querySelector('.product-characteristics__spec-value'));
      if (k) specs[k] = v;
    });
    return specs;
  }

  // характеристики свёрнуты (видно ~14 штук) — раскрываем все
  function expandSpecs() {
    const block = document.querySelector('.product-card__characteristics');
    if (!block || block.classList.contains('product-card__characteristics_placeholder')) {
      block && block.scrollIntoView({ block: 'center' }); // блок грузится лениво
      return false;
    }
    if (document.querySelector('.product-characteristics_collapsed')) {
      const btn = document.querySelector('.product-characteristics__expand');
      btn && btn.click();
      return false;
    }
    return true;
  }

  // товара нет в наличии: пометка в верхнем блоке карточки или в блоке покупки
  function outOfStock() {
    const t = [...document.querySelectorAll('.product-card-top, .product-buy')]
      .map(el => el.innerText).join(' ').toLowerCase();
    return /нет в наличии|нет в продаже|снят с продажи|закончился|временно отсутствует/.test(t);
  }

  function readCard(url) {
    const crumbs = [...document.querySelectorAll('.product-card__breadcrumbs a')]
      .map(a => a.innerText.trim()).filter(Boolean);
    return {
      url,
      code: text(document.querySelector('.product-card-top__code')),
      title: text(document.querySelector('.product-card-top__title')),
      category: crumbs.slice(1).join(' / '),   // без «Каталог»
      description: readDescription(),
      specs: readSpecs(),
    };
  }

  // ссылки со страницы выдачи / каталога
  function saveLinks() {
    const now = [...document.querySelectorAll('.catalog-product a[href*="/product/"]')]
      .filter(a => /\/product\/[0-9a-f]+\//.test(a.pathname)).map(a => clean(a.href));
    const old = load('dns_links', '[]');
    const all = [...new Set([...old, ...now])];
    if (all.length !== old.length) save('dns_links', all);
  }

  function nextLink() {
    const cards = load('dns_cards', '{}');
    const skipped = load('dns_skipped', '[]');
    const left = load('dns_links', '[]').filter(u => !cards[u] && !skipped.includes(u));
    return left[Math.floor(Math.random() * left.length)];   // случайная из оставшихся
  }

  let leaving = false;
  function goNext() {
    if (leaving || !load('dns_auto', 'false')) return;
    if (Object.keys(load('dns_cards', '{}')).length >= TARGET) return save('dns_auto', false);
    const url = nextLink();
    if (!url) return;
    leaving = true;
    const ms = PAUSE[0] + Math.random() * (PAUSE[1] - PAUSE[0]);
    setTimeout(() => (location.href = url), ms);
  }

  // автосохранение открытой карточки
  const tries = {};
  function tick() {
    if (!isProduct()) { saveLinks(); return render(); }
    const url = clean(location.href);
    const cards = load('dns_cards', '{}');
    const skipped = load('dns_skipped', '[]');
    if (cards[url] || skipped.includes(url)) { render(); return goNext(); }
    if (outOfStock()) {              // не в наличии — сразу пропускаем
      skipped.push(url);
      save('dns_skipped', skipped);
      render('нет в наличии — пропуск');
      return goNext();
    }

    tries[url] = (tries[url] || 0) + 1;
    const ready = expandSpecs();
    const card = readCard(url);
    const ok = ready && card.description.length >= MIN_DESC && Object.keys(card.specs).length > 0;

    if (ok) {
      cards[url] = card;
      save('dns_cards', cards);
    } else if (tries[url] >= MAX_TRIES) {
      skipped.push(url);           // нет описания или характеристик — пропускаем
      save('dns_skipped', skipped);
    } else {
      return render(`жду карточку… ${tries[url]}/${MAX_TRIES}`);
    }
    render();
    goNext();
  }

  function download() {
    const data = Object.values(load('dns_cards', '{}'));
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }));
    a.download = `dns_cards_${data.length}.json`;
    a.click();
  }

  function reset() {
    if (!confirm('Удалить все сохранённые карточки и ссылки?')) return;
    ['dns_cards', 'dns_links', 'dns_skipped', 'dns_auto'].forEach(k => localStorage.removeItem(k));
    render();
  }

  // панель
  const box = document.createElement('div');
  box.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:99999;background:#fff;'
    + 'border:1px solid #ccc;border-radius:8px;padding:8px 10px;font:13px sans-serif;'
    + 'box-shadow:0 2px 8px rgba(0,0,0,.2);max-width:260px';
  document.body.appendChild(box);

  function button(label, fn) {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText = 'margin:4px 4px 0 0;padding:2px 6px;cursor:pointer';
    b.onclick = fn;
    return b;
  }

  function render(status = '') {
    const n = Object.keys(load('dns_cards', '{}')).length;
    const links = load('dns_links', '[]').length;
    const skipped = load('dns_skipped', '[]').length;
    const auto = load('dns_auto', 'false');
    box.innerHTML = `<b>DNS cards</b> ${n}/${TARGET}<br>ссылок: ${links} · пропущено: ${skipped}`
      + (status ? `<br><i>${status}</i>` : '');
    box.append(document.createElement('br'),
      button(auto ? '⏸ Стоп' : '▶ Авто', () => {
        save('dns_auto', !auto);
        if (!auto) isProduct() ? tick() : goNext();
        render();
      }),
      button('JSON', download),
      button('Сброс', reset));
  }

  setInterval(tick, 1500);
  tick();
})();
