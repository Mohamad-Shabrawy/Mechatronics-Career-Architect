/**
 * app.js — Mechatronics Career Architect frontend logic
 *
 * Responsibilities:
 *   1. Drag-and-drop + click upload with client-side validation
 *   2. Form submit → POST /api/v1/analyze via fetch (no full page reload)
 *   3. renderResults(data) — builds and animates the results section
 *   4. PDF download — decodes base64 in-browser, no server round-trip
 *
 * All DOM queries use IDs defined in index.html.
 * The AbortController gives the user a 31-second client-side timeout
 * as a safety net on top of the server-side 30-second limit.
 */

// ── Constants ─────────────────────────────────────────────────────────────────
const MAX_FILE_BYTES = 10 * 1024 * 1024;   // 10 MB — mirrors server validation
const FETCH_TIMEOUT_MS = 185_000;           // just over the server 180s limit

// ── DOM references ────────────────────────────────────────────────────────────
const uploadForm    = document.getElementById("upload-form");
const cvFileInput   = document.getElementById("cv_file");
const dropLabel     = document.getElementById("drop-label");
const dropText      = document.getElementById("drop-text");
const submitBtn     = document.getElementById("submit-btn");
const errorMsg      = document.getElementById("error-msg");
const loadingState  = document.getElementById("loading-state");
const loadingText   = document.getElementById("loading-text");
const formPage      = document.getElementById("form-page");
const resultsPage   = document.getElementById("results-page");


// ── Utility helpers ───────────────────────────────────────────────────────────

/** Show an inline error message (red banner above the form). */
function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
  // Re-trigger the shake animation by toggling a class
  errorMsg.style.animation = "none";
  void errorMsg.offsetWidth;          // force reflow
  errorMsg.style.animation = "";
  errorMsg.scrollIntoView({ behavior: "smooth", block: "center" });
}

function clearError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

/** Client-side validation mirrors server rules so the user gets instant feedback. */
function validateFile(file) {
  if (!file) return "Please upload your CV.";
  if (!file.name.toLowerCase().endsWith(".pdf")) return "Only PDF files are accepted.";
  if (file.size > MAX_FILE_BYTES) return "CV file must be under 10MB.";
  return null; // valid
}

/** Switch the submit button into the loading state and show the spinner. */
function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  loadingState.hidden = !isLoading;
  if (isLoading) {
    submitBtn.querySelector(".btn-label").textContent = "PROCESSING...";
  } else {
    submitBtn.querySelector(".btn-label").textContent = "RUN ANALYSIS";
  }
}

/** Cycle the loading status text through descriptive phases so it feels alive. */
let _loadingMsgTimer = null;

function startLoadingMessages() {
  const messages = [
    "PARSING CV DATA",
    "EXTRACTING SKILLS & KEYWORDS",
    "RUNNING NICHE CLASSIFIER",
    "COMPUTING FIT SCORES",
    "ANALYSING SKILLS GAP",
    "RENDERING RADAR CHART",
    "BUILDING CAREER ROADMAP PDF",
    "FINALISING REPORT",
  ];
  let idx = 0;
  // Update every 3.5 seconds — slightly under the average step time
  _loadingMsgTimer = setInterval(() => {
    idx = Math.min(idx + 1, messages.length - 1);
    // Update without the dots-pulse span — we re-add it
    loadingText.innerHTML = `${messages[idx]}<span class="dots-pulse"></span>`;
  }, 3500);
}

function stopLoadingMessages() {
  if (_loadingMsgTimer) {
    clearInterval(_loadingMsgTimer);
    _loadingMsgTimer = null;
  }
  loadingText.innerHTML = `INITIALISING PIPELINE<span class="dots-pulse"></span>`;
}


// ── Drag-and-drop upload zone ─────────────────────────────────────────────────

["dragenter", "dragover"].forEach(evt => {
  dropLabel.addEventListener(evt, e => {
    e.preventDefault();
    dropLabel.classList.add("drop-zone--over");
  });
});

["dragleave", "dragend", "drop"].forEach(evt => {
  dropLabel.addEventListener(evt, e => {
    e.preventDefault();
    dropLabel.classList.remove("drop-zone--over");
  });
});

dropLabel.addEventListener("drop", e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) {
    // Assign the dropped file to the hidden <input> so FormData picks it up
    const dt = new DataTransfer();
    dt.items.add(file);
    cvFileInput.files = dt.files;
    updateDropLabel(file);
    triggerRipple(e);
  }
});

cvFileInput.addEventListener("change", () => {
  const file = cvFileInput.files[0];
  if (file) updateDropLabel(file);
});

function updateDropLabel(file) {
  dropText.innerHTML = `<span class="drop-zone__browse">${escapeHtml(file.name)}</span> selected`;
  dropLabel.classList.add("drop-zone--selected");
  dropLabel.classList.remove("drop-zone--over");
}

/**
 * Ripple visual feedback on file drop.
 * Creates a circular expanding overlay at the drop position.
 */
function triggerRipple(event) {
  const rect   = dropLabel.getBoundingClientRect();
  const x      = (event.clientX || rect.left + rect.width / 2) - rect.left;
  const y      = (event.clientY || rect.top + rect.height / 2) - rect.top;
  const size   = Math.max(rect.width, rect.height);

  const ripple = document.createElement("span");
  ripple.className    = "drop-zone__ripple";
  ripple.style.width  = `${size}px`;
  ripple.style.height = `${size}px`;
  ripple.style.left   = `${x - size / 2}px`;
  ripple.style.top    = `${y - size / 2}px`;

  dropLabel.appendChild(ripple);
  ripple.addEventListener("animationend", () => ripple.remove());
}


// ── Form submission ───────────────────────────────────────────────────────────

uploadForm.addEventListener("submit", async e => {
  e.preventDefault();
  clearError();

  // Client-side validation — gives instant feedback before the network call
  const file = cvFileInput.files[0] || null;
  const fileError = validateFile(file);
  if (fileError) {
    showError(fileError);
    return;
  }

  setLoading(true);
  startLoadingMessages();

  const formData = new FormData(uploadForm);
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const response = await fetch("/api/v1/analyze", {
      method: "POST",
      body:   formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      // Try to parse JSON error from the server; fall back to status text
      let message = `Analysis failed (HTTP ${response.status}).`;
      try {
        const errBody = await response.json();
        message = errBody.message || message;
      } catch (_) { /* body wasn't JSON */ }
      throw new Error(message);
    }

    const data = await response.json();
    window._analysisResult = data;
    stopLoadingMessages();
    setLoading(false);
    transitionToResults(data);

  } catch (err) {
    clearTimeout(timeoutId);
    stopLoadingMessages();
    setLoading(false);

    if (err.name === "AbortError") {
      showError("The analysis is taking too long. Please try again.");
    } else {
      showError(err.message || "An unexpected error occurred. Please try again.");
    }
  }
});


// ── Form ↔ Results transition ─────────────────────────────────────────────────

function transitionToResults(data) {
  // Animate the form out, then reveal results
  formPage.classList.add("page-section--exiting");

  formPage.addEventListener("animationend", () => {
    formPage.hidden = true;
    formPage.classList.remove("page-section--exiting");
    resultsPage.hidden = false;
    resultsPage.classList.add("page-section--entering");
    resultsPage.addEventListener("animationend", () => {
      resultsPage.classList.remove("page-section--entering");
    }, { once: true });

    renderResults(data);

    // Smooth scroll to top of results
    resultsPage.scrollIntoView({ behavior: "smooth", block: "start" });
  }, { once: true });
}


// ── Results renderer ──────────────────────────────────────────────────────────

/**
 * renderResults(data) — populates the #results-page section from API JSON.
 *
 * Called once after a successful /api/v1/analyze POST.
 * Each sub-section is populated independently so a failure in one
 * doesn't block the others.
 */
function renderResults(data) {
  renderPivotNote(data);
  renderEqualScoresNotice(data);
  renderNicheRankings(data.ranked_niches || []);
  renderRadarChart(data.chart_image_b64);
  renderSkillsGap(data.skills_gap || {});
  setupDownloadButton(data);
}

// ── Pivot note ────────────────────────────────────────────────────────────────

function renderPivotNote(data) {
  const pivotNote = document.getElementById("pivot-note");
  const pivotExpl = document.getElementById("pivot-explanation");
  if (data.pivot_applied && data.pivot_explanation) {
    pivotExpl.textContent = data.pivot_explanation;
    pivotNote.hidden = false;
  } else {
    pivotNote.hidden = true;
  }
}

// ── Equal scores notice ───────────────────────────────────────────────────────

function renderEqualScoresNotice(data) {
  const notice = document.getElementById("equal-scores-notice");
  notice.hidden = !data.all_equal_scores;
}

// ── Niche ranking cards ───────────────────────────────────────────────────────

function renderNicheRankings(niches) {
  const container = document.getElementById("niche-cards");
  container.innerHTML = "";

  niches.forEach((niche, index) => {
    const card = buildNicheCard(niche, index);
    container.appendChild(card);

    // Trigger the bar-fill animation after the card's entrance animation completes
    const delay = index * 120; // stagger in ms
    setTimeout(() => {
      const bar = card.querySelector(".niche-bar-fill");
      if (bar) bar.style.width = `${niche.score_pct}%`;
    }, delay + 600); // card entrance + slight pause
  });
}

function buildNicheCard(niche, index) {
  const delay = `${index * 120}ms`;
  const rankClass = index < 3 ? `niche-rank--${index + 1}` : "niche-rank--other";

  const card = document.createElement("div");
  card.className   = "niche-card";
  card.style.setProperty("--delay", delay);
  card.setAttribute("role", "listitem");

  const label = niche.niche_label || formatNiche(niche.niche);
  const pct   = niche.score_pct ?? Math.round((niche.composite_score || 0) * 100);

  card.innerHTML = `
    <div class="niche-rank ${rankClass}" aria-label="Rank ${niche.rank}">
      ${niche.rank}
    </div>
    <div class="niche-info">
      <div class="niche-name">${escapeHtml(label)}</div>
      <div class="niche-bar-track" role="progressbar"
           aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"
           aria-label="${escapeHtml(label)} fit score: ${pct}%">
        <div class="niche-bar-fill" style="--delay:${delay}; width:0%"></div>
      </div>
    </div>
    <div class="niche-right">
      ${index === 0 ? '<span class="top-match-badge" aria-label="Top match">TOP MATCH</span>' : ''}
      <span class="niche-score" aria-hidden="true">${pct}%</span>
    </div>
  `;
  return card;
}

/** Convert "robotics" → "Robotics", "embedded_systems" → "Embedded Systems" */
function formatNiche(niche) {
  return (niche || "")
    .split("_")
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// ── Radar chart ───────────────────────────────────────────────────────────────

function renderRadarChart(chartB64) {
  const img      = document.getElementById("chart-img");
  const fallback = document.getElementById("chart-unavailable");

  if (chartB64) {
    img.src    = `data:image/png;base64,${chartB64}`;
    img.hidden = false;
    fallback.hidden = true;
  } else {
    img.hidden = true;
    fallback.hidden = false;
  }
}

// ── Skills gap ────────────────────────────────────────────────────────────────

function renderSkillsGap(gap) {
  const coveragePct     = Math.round(gap.coverage_percentage || 0);
  const presentSkills   = gap.present_skills || [];
  const missingSkills   = gap.missing_skills || [];

  // Animated counter — counts up from 0 to coveragePct
  animateCounter(
    document.getElementById("coverage-pct"),
    0,
    coveragePct,
    "%",
    1000
  );

  // SVG ring — stroke-dashoffset animates from full (empty) to the coverage value
  const ringFill = document.getElementById("coverage-ring-fill");
  const circumference = 150.796;                    // 2π × r where r=24
  const offset = circumference * (1 - coveragePct / 100);
  // Need to inject the gradient def into the SVG so the ring uses it
  const svg = ringFill.closest("svg");
  if (!svg.querySelector("defs")) {
    svg.insertAdjacentHTML("afterbegin", `
      <defs>
        <linearGradient id="coverage-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#38bdf8"/>
          <stop offset="100%" stop-color="#818cf8"/>
        </linearGradient>
      </defs>
    `);
  }
  // Colour-code ring + counter by coverage level
  const ringColor = coveragePct >= 70 ? '#34D399' : coveragePct >= 40 ? '#FBBF24' : '#F87171';
  ringFill.style.stroke = ringColor;
  document.getElementById('coverage-pct').style.color = ringColor;

  // The transition is CSS-driven; we just set the final value
  requestAnimationFrame(() => {
    ringFill.style.strokeDashoffset = offset;
  });

  // Present skills badges
  const presentList = document.getElementById("present-skills-list");
  presentList.innerHTML = "";
  presentSkills.forEach((skill, i) => {
    presentList.appendChild(buildSkillBadge(skill, "present", i));
  });

  // Missing skills badges
  const missingList  = document.getElementById("missing-skills-list");
  const noMissingMsg = document.getElementById("no-missing-msg");
  missingList.innerHTML = "";

  if (missingSkills.length === 0) {
    noMissingMsg.hidden = false;
  } else {
    noMissingMsg.hidden = true;
    missingSkills.forEach((skill, i) => {
      missingList.appendChild(buildSkillBadge(skill, "missing", i));
    });
  }
}

/** Build a single skill badge <li> with staggered pop animation. */
function buildSkillBadge(skill, type, index) {
  const delay = `${index * 60}ms`;
  const li    = document.createElement("li");
  li.style.setProperty("--delay", delay);
  const badge = document.createElement("span");
  badge.className = `skill-badge skill-badge--${type}`;
  badge.style.setProperty("--delay", delay);
  badge.textContent = skill;
  li.appendChild(badge);
  return li;
}

/**
 * animateCounter — counts a number element from `start` to `end` over `durationMs`.
 * Respects prefers-reduced-motion by jumping straight to the final value.
 */
function animateCounter(element, start, end, suffix, durationMs) {
  const prefersReducedMotion =
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (prefersReducedMotion) {
    element.textContent = `${end}${suffix}`;
    return;
  }

  const startTime = performance.now();
  function tick(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / durationMs, 1);
    // Ease-out cubic
    const eased  = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * eased);
    element.textContent = `${current}${suffix}`;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── PDF download ──────────────────────────────────────────────────────────────

function setupDownloadButton(data) {
  const btn      = document.getElementById("download-btn");
  const errEl    = document.getElementById("download-error");
  errEl.hidden   = true;

  btn.addEventListener("click", () => {
    try {
      const result = window._analysisResult || data;
      if (!result.pdf_b64) throw new Error("No PDF data available");

      // Decode base64 → Uint8Array → Blob — never touches the server again
      const raw   = atob(result.pdf_b64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

      const blob   = new Blob([bytes], { type: "application/pdf" });
      const url    = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href     = url;
      anchor.download = result.pdf_filename || "career_roadmap.pdf";
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      setTimeout(() => URL.revokeObjectURL(url), 10_000);

    } catch (err) {
      console.error("PDF download failed:", err);
      errEl.hidden = false;
    }
  }, { once: true });
}

// ── HTML escaping ─────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
