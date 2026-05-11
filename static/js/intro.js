(function () {
  'use strict';

  // Skip everything for users who prefer no motion.
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const el = document.getElementById('mca-intro');
    if (el) el.remove();
    showLanding(false);
    return;
  }

  const alreadySeen = sessionStorage.getItem('mca-intro-seen') === '1';

  if (alreadySeen) {
    // Return visit — remove overlay immediately, show landing without animation delay.
    const el = document.getElementById('mca-intro');
    if (el) el.remove();
    showLanding(false);
    return;
  }

  // ── First visit: build the intro animation ──────────────────────────────

  // Mark landing page so its reveal animations fire after the intro fades out.
  const lp = document.getElementById('landing-page');
  if (lp) lp.classList.add('lp-animate');

  // Build per-letter spans for the wordmark.
  const wm = document.getElementById('mca-intro-wordmark');
  if (wm) {
    const allChars = [];
    wm.querySelectorAll('.word').forEach(function (w) {
      for (const ch of w.dataset.word) allChars.push({ word: w, ch: ch });
    });
    const startMs  = 1600;
    const windowMs = 1700;
    const total    = allChars.length;
    const easeOut  = function (t) { return 1 - Math.pow(1 - t, 1.6); };
    allChars.forEach(function (entry, i) {
      const span = document.createElement('span');
      span.className = 'ch';
      span.textContent = entry.ch;
      const tt    = i / Math.max(1, total - 1);
      const delay = startMs + windowMs * easeOut(tt);
      span.style.animationDelay = delay + 'ms';
      entry.word.appendChild(span);
    });
  }

  // Spawn 22 drift particles.
  const pc = document.getElementById('mca-intro-particles');
  if (pc) {
    for (let i = 0; i < 22; i++) {
      const s = document.createElement('span');
      s.style.left             = (Math.random() * 100).toFixed(2) + '%';
      s.style.top              = (60 + Math.random() * 30).toFixed(2) + '%';
      s.style.animationDelay   = (Math.random() * 8).toFixed(2) + 's';
      s.style.animationDuration= (6 + Math.random() * 8).toFixed(2) + 's';
      s.style.opacity          = (0.4 + Math.random() * 0.5).toFixed(2);
      s.style.transform        = 'scale(' + (0.6 + Math.random() * 1.4).toFixed(2) + ')';
      pc.appendChild(s);
    }
  }

  // Lock scroll during intro.
  document.body.style.overflow = 'hidden';

  // Trigger niche bar fills after landing page is visible.
  setTimeout(function () {
    if (lp) lp.classList.add('lp-ready');
  }, 8100);

  // Dismount overlay and unlock scroll after animation completes.
  setTimeout(function () {
    const root = document.getElementById('mca-intro');
    if (root) root.remove();
    document.body.style.overflow = '';
    sessionStorage.setItem('mca-intro-seen', '1');
  }, 8400);

  // Click-to-skip: tap anywhere on the overlay to dismiss early.
  const root = document.getElementById('mca-intro');
  if (root) {
    root.style.pointerEvents = 'auto';
    root.addEventListener('click', function () {
      root.remove();
      document.body.style.overflow = '';
      sessionStorage.setItem('mca-intro-seen', '1');
      if (lp) {
        lp.classList.remove('lp-animate');
        lp.classList.add('lp-ready');
      }
    });
  }

  // ── Landing page controls ───────────────────────────────────────────────

  // For return visits, fill bars shortly after load.
  if (alreadySeen && lp) {
    setTimeout(function () { lp.classList.add('lp-ready'); }, 600);
  }

  // CTA and nav-Analyse both transition to the form page.
  ['landing-cta'].forEach(function (id) {
    const btn = document.getElementById(id);
    if (btn) btn.addEventListener('click', goToApp);
  });

  // Replay intro button.
  const replay = document.getElementById('landing-replay');
  if (replay) {
    replay.addEventListener('click', function () {
      sessionStorage.removeItem('mca-intro-seen');
      location.reload();
    });
  }

  function showLanding(animate) {
    const lp = document.getElementById('landing-page');
    if (!lp) return;
    if (!animate) {
      lp.classList.add('lp-ready');
    }
    // Wire up CTA / nav / replay now that landing is live.
    ['landing-cta'].forEach(function (id) {
      const btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', goToApp);
    });
    const replay = document.getElementById('landing-replay');
    if (replay) {
      replay.addEventListener('click', function () {
        sessionStorage.removeItem('mca-intro-seen');
        location.reload();
      });
    }
  }

  function goToApp() {
    const landing = document.getElementById('landing-page');
    const form    = document.getElementById('form-page');
    if (!landing || !form) return;

    // Fade landing out.
    landing.classList.add('lp-exiting');

    // Reveal form — scroll to top first so the form isn't below the fold.
    window.scrollTo(0, 0);
    form.hidden = false;
    form.classList.add('fp-entering');

    setTimeout(function () {
      landing.remove();
    }, 420);

    setTimeout(function () {
      form.classList.remove('fp-entering');
    }, 520);
  }

})();
