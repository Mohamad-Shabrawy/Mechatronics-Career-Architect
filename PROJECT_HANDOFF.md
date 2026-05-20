# Mechatronics Career Architect — Project Handoff

> Read this top to bottom before touching any code. Everything a new session
> needs to understand the project, make changes safely, and avoid breaking things.

---

## What This Project Is

A Flask web application that analyses a mechatronics engineering student's CV (PDF)
and recommends which of six career niches best matches their skills and preferences.
It outputs a ranked niche list, a skills gap breakdown, a proficiency radar chart,
and a downloadable personalised PDF roadmap — all in a single HTTP request.

**The six niches:**
`industrial_automation` · `robotics` · `embedded_systems` · `automotive` · `mechanical_design` · `technical_management`

---

## How to Run It

```bash
# From the project root
python run.py
# → http://127.0.0.1:5002
```

**Port is always 5002.** Ports 5000 and 5001 have permanent zombie sockets on this
machine — never change the port.

The model path is hardcoded in `run.py`:
```python
os.environ.setdefault("MODEL_PATH", "models/20260423_155216_seed42.joblib")
```
The active model artifact is `models/20260423_155216_seed42.joblib`.
Two older artifacts exist: `20260418_005030` and `20260423_133324`. Do not delete them.

---

## How to Run Tests

```bash
python -m pytest tests/ -q
```

**Target: 495 passed, zero failures.** This number must not regress. Any code
change should be followed by a full test run to confirm it.

---

## Project Structure

```
mechatronics-career-architect/
│
├── run.py                        Entry point — starts Waitress WSGI server
├── src/
│   ├── app.py                    Flask application factory (create_app)
│   ├── parser.py                 Phase 1 — PDF text extraction
│   ├── enricher.py               Phase 2 — NER, keywords, 40D feature vector
│   ├── classifier.py             Phase 3 — ML niche classifier
│   ├── scorer.py                 Phase 4 — hybrid fit scorer (60/40 formula)
│   ├── gap_analyzer.py           Phase 5a — skills gap analysis
│   ├── visualizer.py             Phase 5b — radar chart PNG (matplotlib Agg)
│   ├── report_generator.py       Phase 5c — career roadmap PDF (reportlab)
│   ├── model_io.py               Model artifact I/O + CANONICAL_NICHES list
│   ├── feature_vector.py         VECTOR_INDEX_ORDER — the 40D schema definition
│   ├── lexicon.py                Mechatronics keyword lexicon
│   ├── synonyms.py               Keyword synonym expansion
│   ├── semantic_map.py           Semantic dimension groupings
│   ├── entity_types.py           NER entity type definitions
│   ├── dataset.py                Training dataset loader
│   ├── trainer.py                Model training script
│   └── api/v1/routes.py          API blueprint — POST /api/v1/analyze, GET /api/v1/health
│
├── templates/index.html          Single HTML file for the entire frontend
├── static/
│   ├── css/style.css             Main stylesheet (v14)
│   ├── css/intro.css             Cinematic intro overlay styles
│   ├── js/app.js                 Frontend logic (form submit, results render)
│   └── js/intro.js               Intro animation + landing page controller
│
├── config/
│   └── scoring_config.yaml       Intent affinity map + override penalty rules
│
├── data/
│   ├── niche_benchmarks.json     Top-10 skill names per niche (human-readable)
│   └── training/dataset.json     Synthetic training dataset (2400 CVs)
│
├── models/                       Trained model artifacts (.joblib)
├── logs/predictions.jsonl        Prediction log (timestamp, hash, niche, confidence)
└── tests/                        495 tests across unit, integration, contract
```

---

## The Pipeline — Phase by Phase

Every CV upload goes through `run_pipeline()` in `src/api/v1/routes.py`.
The function runs inside a **180-second timeout** enforced by a thread executor.

```
PDF bytes
  ↓
Phase 1  parse_cv(tmp_path)           → {skills, projects, education, internships}
  ↓
Phase 2  enrich_cv(phase1_result)     → {feature_vector: [40 floats], entities, cluster_profile}
  ↓
Phase 3  predict_niche(fv, artifact)  → {predicted_niche, confidence}   [informational only]
  ↓
Phase 4  compute_fit_score(fv, artifact, prefs) → {ranked_niches: [...6...], fallback_used, pivot_applied}
  ↓
Phase 5a analyze_skills_gap(fv, niche, artifact) → {present: [...], missing: [...]}
           (runs for top 3 niches, not just rank-1)
  ↓
Phase 5b generate_radar_chart(fit_score_result)  → {chart_image_bytes: PNG bytes}
  ↓
Phase 5c generate_career_roadmap_pdf(...)         → {pdf_bytes, filename}
  ↓
JSON response to browser
```

**Critical:** Phase 3's prediction is included in the response JSON but is NOT
used by Phase 4. The scorer (Phase 4) works directly from the feature vector and
the niche centroids stored in the model artifact — it is independent of the
classifier's top prediction.

**Privacy constraint (hard rule):** All processing is in-memory only. No pipeline
result is ever written to disk. The only disk write allowed is the prediction log
(`logs/predictions.jsonl`), which stores a truncated SHA-256 hash of the feature
vector — never the vector itself or any CV content.

---

## Phase 4 Scoring Formula

```
For each niche:
    cosine_score = cosine_similarity(feature_vector, niche_centroid)
                   → 0.0 when feature_vector is all-zeros (fallback_used: true)
    intent_score = clamp(0.5 + sum(affinity_deltas), 0.0, 1.0)
    composite    = 0.60 × cosine_score + 0.40 × intent_score
    final        = composite − override_penalty   (if any rule triggers)

Normalize: divide all scores by the winner's score → winner always = 1.0
```

**When all sliders are 0 (neutral):** every niche gets intent = 0.5 baseline,
so the 40% intent component cancels out and ranking is driven entirely by the CV.

**Override penalties** are post-composite subtractions from `config/scoring_config.yaml`.
They trigger only at extreme slider values (±2). Max penalty is 0.25, enough to
demote a niche by at least one rank even if it led by 0.24 in composite score.

**`scoring_config.yaml` is the single source of truth** for all affinity weights
and override rules. It can be tuned without touching Python code. If this file is
missing, Phase 4 returns `CONFIG_ERROR` and the entire pipeline fails.

---

## The Feature Vector

`src/feature_vector.py` defines `VECTOR_INDEX_ORDER` — a fixed 40-element sequence
of `(entity_type, cv_section)` pairs. This is the schema the model was trained on.

A SHA-256 hash of this order is embedded in every saved model artifact. `model_io.py`
checks this hash at load time and raises `SchemaVersionError` if there is a mismatch.
**Never change `VECTOR_INDEX_ORDER`** unless you also retrain and save a new artifact.

---

## The Model Artifact

Loaded from `models/20260423_155216_seed42.joblib` at app startup via `joblib`.
Structure (dict):
```python
{
  "model":          sklearn RandomForestClassifier,
  "centroids":      {niche_name: np.array(40,)},   # mean FV per niche
  "niche_top_skills": {niche_name: [int, ...]},     # top 10 dimension indices
  "schema_version": str,                             # hash of VECTOR_INDEX_ORDER
  "metadata":       {trained_at, seed, ...}
}
```

The model is loaded **once** at startup inside `create_app()` and stored in
`app.config["MODEL_ARTIFACT"]`. It is passed into `run_pipeline()` on every request.

---

## API Endpoints

### `POST /api/v1/analyze`
Accepts `multipart/form-data`:
- `cv_file` — PDF, max 10 MB
- `work_environment`, `system_level`, `industry_interest`, `team_scale`, `travel_tolerance` — integers in `[-2, 2]`

Returns JSON:
```json
{
  "ranked_niches":     [...],
  "all_equal_scores":  false,
  "pivot_applied":     false,
  "pivot_explanation": null,
  "skills_gap":        {...},
  "chart_image_b64":   "...",
  "pdf_b64":           "...",
  "pdf_filename":      "career_roadmap.pdf"
}
```

Error shapes always include `{"error": true, "message": "...", "code": "..."}`.
Never expose raw exceptions or stack traces to the client.

### `GET /api/v1/health`
Returns `{"status": "ok", "model_loaded": true/false}`.

---

## Frontend Architecture

The single `templates/index.html` file contains four sequential sections
controlled by `static/js/intro.js` and `static/js/app.js`:

```
1. #mca-intro       Cinematic intro overlay (shown once per session)
                    → typing animation, orbit rings, boot progress bar
                    → removed from DOM by intro.js when complete

2. #landing-page    Marketing landing page
                    → hex grid background, niche preview panel
                    → "Upload CV · Run Analysis" CTA transitions to #form-page

3. #form-page       Upload form (hidden until CTA clicked)
                    → split layout: hero panel (left) + form panel (right)
                    → POST to /api/v1/analyze via fetch
                    → shows loading state during analysis

4. #results-page    Results dashboard (hidden until analysis completes)
                    → niche ranking cards, radar chart, skills gap, PDF download
```

**All HTML IDs are used by app.js.** Never rename or remove them.
Key IDs: `upload-form`, `cv_file`, `submit-btn`, `error-msg`, `loading-state`,
`loading-text`, `form-page`, `results-page`, `niche-cards`, `chart-img`,
`present-skills-list`, `missing-skills-list`, `coverage-pct`, `coverage-ring-fill`,
`download-btn`, `pivot-note`, `pivot-explanation`, `equal-scores-notice`.

---

## Design System

**Aesthetic:** Premium dark sci-fi / engineering HUD. Think OLED black with
neon cyan and electric blue accents. NOT purple on white.

**Fonts:**
- `'DM Sans'` — UI prose, labels, body text
- `'Share Tech Mono'` — data readouts, legends, monospace elements, chip badges

**Colour tokens (CSS variables in style.css):**
```
--bg:          #030712   deep OLED black
--bg-2:        #0B1120
--bg-elevated: #0F172A
--blue:        #60A5FA   sky blue (primary accent)
--blue-mid:    #3B82F6
--blue-dark:   #2563EB
--cyan:        #22D3EE   neon cyan (secondary accent)
--emerald:     #34D399   present skills / success states
--red:         #F87171   missing skills / error states
--amber:       #FBBF24   rank-1 niche highlight
--purple:      #A78BFA   rank-3 accent
--indigo:      #818CF8   gradient midpoint
```

**Active visual effects in style.css (v14):**
- `body::before` — dot-grid circuit background (56px, fades at edges)
- `body::after` — slowly drifting orthogonal circuit trace lines
- `.results-panel::after` — CRT scanline overlay on every result card
- `rank1-pulse` keyframe — amber left-border pulse on the top niche card
- `logo-float`, `ring-spin` — animated SVG logo on form page hero
- Parallax 3D tilt on hero panel and result panels (inline JS in index.html)
- Cinematic intro animation (intro.css + intro.js)

---

## Key Constraints — Never Violate These

| Constraint | Reason |
|---|---|
| No disk writes inside the pipeline | Privacy — CVs are personal data |
| Port 5002 only | Ports 5000/5001 have zombie sockets on this machine |
| Python 3.11+ syntax only | Runtime is Python 3.11 |
| 495 tests must stay green | Any regression is a bug |
| Never retrain the model without explicit instruction | Active artifact is production-calibrated |
| Never change `VECTOR_INDEX_ORDER` without retraining | Hash mismatch = `SchemaVersionError` at startup |
| Never expose raw exceptions to the client | Security + UX rule |
| All HTML IDs in index.html must be preserved | app.js depends on every one of them |
| Matplotlib must use `Agg` backend before pyplot import | Server has no display — any other backend crashes |

---

## Config File Locations

| File | Purpose |
|---|---|
| `config/scoring_config.yaml` | Intent affinity weights + override penalty rules for Phase 4 |
| `data/niche_benchmarks.json` | Human-readable top-10 skill names per niche (used by gap analyzer) |
| `data/training/dataset.json` | 2400-row synthetic training dataset |
| `logs/predictions.jsonl` | Append-only prediction log (hash, niche, confidence, timestamp) |

---

## Common Tasks — What to Touch

| Task | Files to edit |
|---|---|
| Tune niche affinity weights | `config/scoring_config.yaml` only |
| Change UI colours or animations | `static/css/style.css` and/or `static/css/intro.css` |
| Change frontend behaviour | `static/js/app.js` and/or `static/js/intro.js` |
| Change HTML layout | `templates/index.html` — keep all IDs intact |
| Add an API field to the response | `src/api/v1/routes.py` → `analyze()` route only |
| Change skill benchmark names | `data/niche_benchmarks.json` |
| Retrain the model | `src/trainer.py` → `python -m src.trainer` → update MODEL_PATH in `run.py` |
| Add a test | `tests/unit/`, `tests/integration/`, or `tests/contract/` |

---

## What Was Completed Last Session

- CSS/HTML redesign with cyberpunk visual effects (style.css v14)
  - Dot-grid + circuit trace background
  - CRT scanline overlay on result panels
  - Pulsing amber border on rank-1 niche card
  - Blinking terminal cursor on hero title
- `MEMBER5_STUDY_GUIDE.md` — presentation prep doc for the team member
  who built the scoring/gap/visualizer components
