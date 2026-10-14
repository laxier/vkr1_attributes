// DNS: прокрутить весь список выдачи, в DOM оставлять только последние KEEP карточек
(() => {
  const KEEP = 150;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const s = window.__dnsScroll = { done: false, stall: 0, prev: -1, removed: 0, seen: 0 };

  document.getElementById('base-footer')?.style.setProperty('display', 'none');

  (async () => {
    while (!s.done) {
      const cards = [...document.querySelectorAll('.catalog-product')];
      if (cards.length > KEEP) {
        cards.slice(0, cards.length - KEEP).forEach(c => {
          c.querySelectorAll('img').forEach(i => i.removeAttribute('src'));
          c.remove();
          s.removed++;
        });
      }
      const all = document.querySelectorAll('.catalog-product');
      all[all.length - 1]?.scrollIntoView({ block: 'end' });
      window.scrollBy(0, 500);
      await sleep(1500);

      s.seen = document.querySelectorAll('.catalog-product').length + s.removed;
      if (s.seen === s.prev) {
        s.stall++;
        window.scrollBy(0, -1500);
        await sleep(600);
        if (s.stall >= 12) s.done = true;
      } else s.stall = 0;
      s.prev = s.seen;
    }
    console.log('DNS scroll done:', s.seen, 'cards');
  })();
})();
// остановить: window.__dnsScroll.done = true
// прогресс:   window.__dnsScroll
