# PROJECT REPORT BRIEF — Mechatronics Career Architect
### For: Claude Chat → Generate Full Academic Word Report

---

## COVER PAGE INFORMATION

- **University:** Helwan National University
- **Department:** Mechatronics and Robotics Engineering
- **Subject:** Machine Learning
- **Course Code:** CSC2308
- **Professor:** Dr. Hadeer Ahmed
- **Teaching Assistant:** Eng. Asmaa Ibrahim
- **Project Title:** Mechatronics Career Architect — AI-Powered Career Niche Analysis System
- **Academic Year:** 2025–2026

**Team Members:**
| Name | Student ID |
|------|------------|
| Mohamad Sherif Shabrawy | 922230084 |
| Yousef Ahmed Elbeltagy | 922230111 |
| Aley Mohamed Yasser | 922230063 |
| Engy Aly Sayed | 922230009 |
| Amr Sherif Maher | 922230070 |

---

## 1. PROJECT OVERVIEW

### 1.1 Problem Statement

Mechatronics engineering is an unusually wide discipline — a graduating student may be equally competent in embedded systems, robotics, automotive systems, industrial automation, mechanical design, or technical management, yet most students lack a structured, data-driven method to determine which niche best aligns with their actual accumulated skills.

Traditional career advice relies on subjective self-assessment or generic job-board browsing. Neither method can accurately map a student's CV against the real skill profiles demanded by each engineering niche, nor can they quantify the gap between where the student is and where they need to be.

The result is misalignment: students apply to the wrong roles, receive fewer offers, and take longer to establish a career trajectory. The problem is particularly acute at the transition from final year to first employment, where every student competes with peers who have nearly identical academic backgrounds.

### 1.2 Objectives

1. Build a machine learning classifier trained on 2,400 labeled CV profiles to predict the most suitable engineering niche from a student's CV with at least 80% overall accuracy.
2. Implement a hybrid scoring system that blends CV-derived skill signals (60%) with the student's stated work preferences (40%) to produce a ranked list of all six niches.
3. Deliver a quantitative skills gap analysis that identifies which specific technologies a student is missing relative to the benchmark profile for their top recommended niche.
4. Package the entire system as a deployable web application that any student can use by uploading a PDF CV and receiving results in under 3 minutes.
5. Achieve a minimum per-niche F1 score of 0.60 across all six niches to ensure no engineering specialisation is systematically disadvantaged.

### 1.3 Motivation

The mechatronics field is growing rapidly across automotive electrification, robotics, smart manufacturing (Industry 4.0), and embedded IoT systems. Employers in these areas increasingly filter candidates by niche-specific technical keywords and tool experience, not by general engineering degree. A student who has done robotics-flavored coursework but applies exclusively to automotive positions will be screened out — not because they lack potential, but because their CV does not surface the right signals.

This tool was motivated by a simple observation: the data to make this decision correctly already exists inside the student's CV — it just needs to be extracted, encoded, and matched against a learned profile. Machine learning makes this tractable at scale.

---

## 2. THE SIX CAREER NICHES

The system classifies students into one of six canonical Mechatronics engineering niches:

| Niche | Description |
|-------|-------------|
| **Robotics** | Robot kinematics, ROS/ROS2, motion planning, sensor fusion |
| **Embedded Systems** | Microcontrollers, RTOS, firmware, communication protocols |
| **Automotive** | AUTOSAR, CAN/LIN/FlexRay, ISO 26262 functional safety, ADAS |
| **Industrial Automation** | PLC programming, SCADA, HMI, factory automation |
| **Mechanical Design** | CAD/CAM (SolidWorks, CATIA), FEA simulation, GD&T |
| **Technical Management** | Project management, systems engineering, Agile/PMBOK frameworks |

---

## 3. DATASET

### 3.1 Description

- **Dataset name:** Mechatronics Career Architect Training Set (MCA Egypt Edition)
- **Final file:** `training_set.csv` → converted to `dataset.csv` for model training
- **Total samples:** 2,400
- **Class distribution:** Perfectly balanced — exactly 400 entries per niche
- **Columns:** `label` (niche name) + `cleaned_text` (preprocessed engineering text)
- **Feature count:** 40 numerical features per sample (derived from cleaned_text — see Section 3.3)
- **Null values:** 0
- **Duplicates:** 0
- **Final noise rate:** 0.00% (verified by 4 independent audit passes)
- **Real data:** ~1,659 entries sourced from real job listings and resumes
- **Synthetic data:** ~741 entries generated to fill under-represented niches

**Raw source files:**

| File | Rows | Type | Notes |
|------|------|------|-------|
| `Wuzzuf_Jobs.csv` | 4,380 | Job listings | Scraped from Wuzzuf.com, Egypt's largest job platform. Most Egypt-specific source — real company names, locations, local job titles |
| `Cleaned_Wuzzuf_Jobs.csv` | 7,866 | Job listings | Pre-cleaned version of above |
| `job_dataset.csv` | 1,068 | Job descriptions | From Kaggle. Rich in technical detail but mixed with non-engineering roles requiring heavy filtering |
| `SaveData4.csv` | 3,372 | Job title + skills pairs | From regional job boards. Compact but high-signal after filtering |
| `AIbased_Career_Recommendation_System.csv` | 200 | Candidate profiles | Skill sets + recommended careers. Small but useful for skill-to-role association |
| `joblisting.csv` | Variable | Supplementary listings | Variable column structure; used as additional signal |
| Resume archives | 274 files | Resumes | `.pdf`-extension files that were actually ZIP archives containing OCR-extracted `.txt` pages and `.jpeg` image scans. Required a custom ZIP extractor — standard PDF parsers (pdfplumber) returned empty text on all 274 files |

**Domain knowledge backbone:** `mechatronics_keyword_weights.json` — 261 Mechatronics keywords, each with a weight (0.0–1.0) across all 6 niches. Every filtering and classification decision in the pipeline was driven by this file.

### 3.2 Data Collection

**Text extraction per source type:**

- **CSV files:** Relevant columns (job title, description, required skills, responsibilities) were concatenated per row into a single text string
- **Resume archives:** Each `.pdf`-named file was opened as a ZIP archive using Python's `zipfile` module. The `.txt` entries (OCR-extracted resume pages) were extracted and joined in order. A standard PDF reader failed silently on all 274 files because they were not valid PDFs

**3-Layer Purity Gate (filtering non-engineering content):**

Raw sources were heavily contaminated with nurses, accountants, JavaScript developers, and digital marketers. A three-layer filter was applied sequentially to every entry:

**Layer 1 — Blacklist (32 regex patterns)**
Instant rejection if the raw text matched any known non-engineering role indicator. Examples of rejected patterns: `"blockchain developer"`, `"nursing"`, `"solidity"`, `"wordpress"`, `"digital marketing"`, `"penetration test"`. This ran before scoring to avoid wasted compute.

**Layer 2 — Keyword Score Gate**
Each surviving entry was scored against `mechatronics_keyword_weights.json`. To pass, an entry needed EITHER:
- ≥ 2 high-weight keyword hits (weight ≥ 0.7) in its best-matching niche, OR
- A cumulative niche score of ≥ 1.5

This eliminated vague matches that had only generic engineering terms.

**Layer 3 — Dominant Keyword Guard**
Even after Layer 2, the entry had to contain at least ONE keyword scoring ≥ 0.8 in its assigned niche. This blocked generic developers who stacked shared terms (Python, C, Git, Linux) to pass Layer 2 without any niche-specific specialist keywords.

**Text cleaning (applied to stored version only):**
1. Strip all HTML tags
2. Remove non-ASCII characters (Arabic script, special characters)
3. Remove punctuation except `+ # - /` (preserved for C++, I2C, C#, etc.)
4. Remove English stopwords
5. Suffix lemmatisation: `"programming"` → `"program"`, `"automation"` → `"automat"`, `"designing"` → `"design"`

**Critical design decision:** Scoring always ran on **raw text**. Cleaning was applied only to the version stored in the dataset. Applying lemmatisation before keyword matching broke multi-word phrases — `"Mechanical Design"` became `"mechan design"` which no longer matched the dictionary key. This bug was discovered in Version 1 of the pipeline and corrected in Version 2.

**Classification:** Each cleaned entry was labeled with one of the 6 canonical niche labels based on which niche scored highest under the keyword weight matrix.

**Deduplication:** Every entry was MD5-hashed before insertion. If the same cleaned text appeared twice across different source files, the second copy was dropped.

**Balancing to 400 entries per niche:**
- *Over-represented niches* (e.g., Embedded Systems): Only the 400 entries with the highest keyword density (count of distinct matrix keywords per entry) were kept — the richest and most informative
- *Under-represented niches* (e.g., Industrial Automation — Egyptian PLC job posts are typically brief): Synthetic entries were generated from 12 hand-authored templates per niche. Templates covered both industry phrasing (`"Configure PROFINET IO fieldbus topology"`) and student phrasing (`"3rd year graduation project, PLC-based sorting conveyor, Egypt university"`). Every synthetic entry was validated by re-running it through the keyword scorer to confirm it would be correctly classified

**Pipeline iteration history:**

| Version | Problem Discovered | Fix Applied |
|---------|--------------------|-------------|
| v1 | Lemmatiser ran before keyword matching — "Mechanical Design" → "mechan design" → zero score. All 274 resume files returned empty (PDF reader on ZIP archives). Industrial Automation and Mechanical Design had near-zero real entries | Separated score-mode (raw text) from store-mode (cleaned text). Rewrote resume extractor using `zipfile` module |
| v2 | Noise audit: 14.6% contamination (350/2,400 entries were blockchain developers, data scientists, web developers who stacked shared keywords) | Introduced Blacklist (Layer 1) and Dominant Keyword Guard (Layer 3) |
| v3 | Layer 3 threshold too strict — only 114 real entries survived across all 6 niches. Legitimate engineering resumes with abbreviated skill lists were rejected | Loosened threshold; added OR logic in Layer 2; kept Layer 3 as supplement not primary gate |
| v4 (final) | Residual contamination in Automotive and Embedded Systems from entries that scored their niche using only the letter "C" as a keyword (C scores 0.9 in Embedded Systems) | Surgical cleanup removed these entries; replaced with synthetic equivalents |

**Formal validation checks on the final dataset:**

| Check | Method | Result |
|-------|--------|--------|
| Row count | Count per label | 2,400 total, exactly 400 per niche ✓ |
| Null values | Pandas `.isnull().sum()` | 0 ✓ |
| Duplicates | MD5 hash comparison | 0 ✓ |
| Noise audit | 28-term blacklist vocabulary scan | 0 entries flagged, 0.00% noise ✓ |
| Keyword coverage | Top 10 dominant keywords (weight ≥ 0.8) each appear in ≥ 15% of their niche's entries | All 60 dominant keywords passed; most appeared in > 60% ✓ |
| Minimum entry length | ≥ 8 tokens after cleaning | All 2,400 entries passed ✓ |

The noise audit was run 4 separate times: after v2 (14.6% found), after v3 (over-filtered), after v4 (residual found), and after the surgical fix (0.00% confirmed).

### 3.3 Data Preprocessing

#### Phase 1 — PDF Parsing & Layout Analysis

Before any machine learning can occur, a student's CV must be read correctly from its raw PDF format. This is non-trivial because CVs come in two structural layouts — single-column and two-column — and a naive text extractor reads a two-column CV left-to-right across both columns simultaneously, destroying the section structure entirely.

The parsing pipeline (implemented in `src/parser.py` using PyMuPDF) works in five steps:

**Step 1 — Bounding Box Extraction**

PyMuPDF's `page.get_text("dict")` mode is used instead of plain text extraction. This returns every text block with its exact `(x0, y0, x1, y1)` bounding box coordinates in page points. Capturing position data is the prerequisite for all subsequent layout analysis. Image blocks (type 1) are silently discarded; only text blocks (type 0) are retained.

**Step 2 — Two-Column Layout Detection**

For each page, the parser inspects the `x0` (left edge) coordinate of every text block. If any block starts beyond 40% of the page width, the entire page is classified as two-column. This 40% threshold works because:
- On a standard A4 page (595 points wide), 40% ≈ 238pt
- Right-column content in a two-column CV typically starts at ~300pt (the midpoint)
- Single-column CVs place all text near the left margin (x0 < 100pt), well under 40%

**Step 3 — Reading Order Reconstruction**

For single-column pages: blocks are sorted top-to-bottom by their `y0` coordinate.

For two-column pages: blocks are split at the exact page midpoint. Left-column blocks (x0 < midpoint) are sorted by `y0` and emitted first, then right-column blocks are sorted and emitted. This ensures a full column is read before the other starts, preserving the human reading order.

**Step 4 — Section Header Detection**

The parser walks the ordered blocks and identifies CV section headers using a synonym expansion dictionary (`src/synonyms.py`). For example, all of the following map to the canonical `skills` section:

> "Technical Skills", "Core Competencies", "Skills Summary", "Key Skills", "Technologies"

Three guards prevent false positives:
1. Lines are normalized (lowercased, stripped, trailing colon removed) before matching
2. Lines longer than 60 characters are immediately rejected — section headers are always short
3. Only exact string matches are accepted — no substring matching

**Step 5 — In-Block Header+Content Splitting**

When a text block's first line is a section header and subsequent lines contain content (e.g., a block reading `"SKILLS\nPython, MATLAB, ROS"`), the parser splits them: the header switches the active section, and the remaining lines are appended as content to that section. This prevents content from being silently discarded when a PDF renderer places headers and their content in the same block.

**Keyword Matching (Phase 1 — `_recognize_keywords`)**

Each content block is scanned for Mechatronics keywords from `src/lexicon.py` using case-insensitive regex with custom word-boundary assertions:

```python
pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
```

The `(?<!\w)` / `(?!\w)` lookbehind/lookahead are used instead of standard `\b` because some keywords contain special characters (parentheses, hyphens, dots) where `\b` fails. For example, `\b` incorrectly matches "python" inside "micropython"; the negative lookbehind correctly rejects it. All matches are stored as a deduplicated, alphabetically sorted list for downstream determinism.

---

#### Phase 2 — Named Entity Recognition (NER) and Feature Engineering

Phase 2 (`src/enricher.py`) takes the structured section text from Phase 1 and performs domain-specific Named Entity Recognition using the `ENTITY_TAXONOMY` — a curated dictionary of ~200 Mechatronics engineering terms organized into 10 entity types.

**Multi-Type Entity Handling**

Some terms belong to multiple entity types. For example:
- `"matlab"` → both `programming_language` AND `simulation_tool`
- `"autosar"` → both `automotive_standard` AND `embedded_systems` (via semantic map)

When a term is found, one entity record is emitted *per type* it belongs to. This means a CV heavily featuring MATLAB correctly increments both the programming and simulation dimensions of the feature vector.

**Section-Level Deduplication**

Within each section, each `(term, type)` pair is counted exactly once regardless of how many blocks it appears in. A student who lists "SolidWorks" in five bullet points under Skills still gets a count of 1 for `(solidworks, cad_software, skills)` — not 5. This prevents verbose CVs from artificially inflating their scores over terse but equally skilled CVs.

**Semantic Cluster Profiling**

Beyond the 40D feature vector, Phase 2 builds a secondary `cluster_profile` — a 7-key frequency count that maps each detected entity to a career niche cluster via `src/semantic_map.py`. This profile gives a quick human-readable overview of which niches the CV's entities belong to. Multi-cluster entities (e.g., AUTOSAR belongs to both `embedded_systems` and `automotive`) increment multiple counters. Entities not in the semantic map are counted under `unclassified` rather than dropped.

The cluster profile is not consumed by the ML model (which uses the feature vector only) but is included in the API response for transparency — it shows the student why the system reached its conclusion.

---

#### Entity Type Taxonomy (10 categories)

The raw CV text is not fed directly to the model. Instead, it is processed through a two-stage NLP pipeline that extracts and encodes semantic entities:

| # | Entity Type | Example Terms |
|---|-------------|---------------|
| 1 | `plc_hardware` | Siemens S7, Allen Bradley, Schneider PLC, SCADA, HMI |
| 2 | `cad_software` | SolidWorks, AutoCAD, CATIA, Fusion 360, ANSYS |
| 3 | `microcontroller` | STM32, Arduino, ESP32, PIC, Raspberry Pi |
| 4 | `communication_protocol` | CAN, I2C, SPI, UART, Modbus, PROFINET |
| 5 | `programming_language` | C, C++, Python, MATLAB, LabVIEW |
| 6 | `simulation_tool` | Simulink, MATLAB, Gazebo, Webots, Adams |
| 7 | `robotic_framework` | ROS, ROS2, MoveIt, Nav2, OpenCV |
| 8 | `automotive_standard` | AUTOSAR, ISO 26262, MISRA C, CAN, Vector CANalyzer |
| 9 | `mechanical_tool` | SolidWorks, CATIA, GD&T, FEA, Ansys Mechanical |
| 10 | `management_methodology` | Agile, Scrum, PMBOK, PMP, Jira, MS Project |

#### Feature Vector Construction (40 Dimensions)

Each CV is converted into a **40-dimensional integer feature vector** using a cross-product encoding:

```
Feature Index = (type_index × 4) + section_index

type_index:    0–9  (the 10 entity types above, in fixed order)
section_index: 0=Skills, 1=Projects, 2=Education, 3=Internships
```

Each dimension counts the number of **distinct** entities of that type found in that CV section. For example:
- `feature[0]` = count of distinct PLC hardware entities in the Skills section
- `feature[1]` = count of distinct PLC hardware entities in the Projects section
- `feature[20]` = count of distinct simulation tool entities in the Skills section

This encoding captures not just *what* a student knows, but *where* in their CV it appears — a SolidWorks mention under Skills signals deeper expertise than the same term mentioned inside an Education description.

#### Normalization Notes
- Keyword matching uses whole-word case-insensitive regex with synonym expansion (e.g., "Arduino Uno" → `microcontroller`)
- Deduplication at inference time: each (term, type, section) triple in a student's CV is counted once only per section
- Section detection uses header synonym expansion (see Phase 1 detail above)
- No feature scaling applied — Random Forest is scale-invariant by design

---

## 4. METHODOLOGY

### 4.1 Machine Learning Approach

**Algorithm: Random Forest Classifier (scikit-learn `RandomForestClassifier`)**

Random Forest was chosen over alternative algorithms for the following reasons:

1. **Interpretability:** Feature importance scores are a native output, making it easy to explain which skill categories drive each niche prediction — critical for a student-facing tool.
2. **Robustness to sparse inputs:** Many student CVs have zeros in most of the 40 dimensions. Random Forest handles sparse feature spaces well without requiring imputation.
3. **No scaling requirement:** Integer count features feed directly into the model without normalization.
4. **Ensemble stability:** With 200 trees, the classifier's predictions are stable across different random subsets of the data, reducing variance on small or unusual CVs.

Alternatives considered:
- **SVM:** Rejected due to sensitivity to feature scaling and lower interpretability.
- **Neural Network (MLP):** Rejected due to dataset size (2,400 samples is insufficient to reliably train a deep architecture without overfitting).
- **Logistic Regression:** Considered but rejected because the relationship between skill counts and niche membership is non-linear (a student with high robotic_framework AND high plc_hardware is unusual and needs non-linear decision boundaries).

### 4.2 Model Architecture

**Random Forest Configuration:**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `n_estimators` | 200 | Enough trees for stable ensemble predictions at this dataset size |
| `max_depth` | None (unlimited) | Regularization via `min_samples_leaf` instead of depth cap |
| `min_samples_leaf` | 2 | Prevents single-sample leaf nodes that overfit noise |
| `class_weight` | `balanced` | Equal importance across all 6 niches regardless of any class imbalance |
| `random_state` | 42 | Fixed seed for full reproducibility |

**Hybrid Scoring Layer (post-classification):**

The ML classifier output is augmented by a hybrid scoring formula:

```
composite_score(niche) = 0.60 × cosine_similarity(cv_vector, niche_centroid)
                       + 0.40 × intent_component(user_preferences)
                       − override_penalty(user_preferences, niche)

normalized_score(niche) = composite_score(niche) / max(composite_score for all niches)
```

- **CV Component (60%):** Cosine similarity between the student's 40D feature vector and the niche's centroid vector (the mean feature vector of all training samples for that niche). Measures directional alignment of skill profiles.
- **Intent Component (40%):** Derived from 5 preference dimensions the student optionally provides (work environment, system level, industry interest, team scale, travel tolerance). Starts at 0.5 (neutral) and shifts based on affinity deltas from a YAML configuration.
- **Override Penalty:** Additive post-composite subtraction when a stated preference strongly conflicts with a niche's work context (e.g., strong preference for office work reduces the automotive field score).
- **Max-Normalization:** Top niche always reads as 1.0; all others are expressed as a fraction relative to it.

### 4.3 Tools & Environment

| Category | Tool / Library | Version |
|----------|---------------|---------|
| Language | Python | 3.11+ |
| ML framework | scikit-learn | ≥1.4 |
| Numerical computation | NumPy | ≥1.26 |
| Data manipulation | Pandas | ≥2.1 |
| PDF parsing | PyMuPDF (fitz) | ≥1.23 |
| Web framework | Flask | ≥3.0 |
| WSGI server | Waitress | ≥3.0 |
| Model persistence | joblib | ≥1.3 |
| Chart generation | Matplotlib | ≥3.8 |
| PDF report generation | ReportLab | ≥4.0 |
| Config management | PyYAML | ≥6.0 |
| Testing | pytest | ≥8.0 |
| Development OS | Windows 11 |  |

---

## 5. IMPLEMENTATION

### 5.1 System Design

The system is structured as a **6-phase sequential pipeline**, each phase consuming the output of the previous:

```
Phase 1: PDF → Structured Text (NLP Extraction)
           ↓
Phase 2: Structured Text → 40D Feature Vector (Semantic NER)
           ↓
Phase 3: Feature Vector → Niche Probabilities (ML Classifier)
           ↓
Phase 4: Probabilities + Preferences → Ranked Scores (Hybrid Scorer)
           ↓
Phase 5: Ranked Scores + Feature Vector → Gap Analysis + Visualizations
           ↓
Phase 6: All outputs → Web UI / PDF Report (Frontend + API)
```

**Key design decisions:**
- **In-memory only:** No CV data, predictions, or charts are written to disk at any point. All processing happens in RAM for privacy.
- **Stateless API:** Each HTTP request runs the full pipeline independently. No session state is stored server-side.
- **Model warmup:** The trained `.joblib` artifact and all heavy libraries (NumPy, Matplotlib, ReportLab) are pre-loaded at server startup so the first user request is not penalized by import latency.

**File structure:**
```
mechatronics-career-architect/
├── src/
│   ├── parser.py          # Phase 1 — PDF parsing + section detection
│   ├── lexicon.py         # Phase 1 — Mechatronics keyword dictionary
│   ├── synonyms.py        # Phase 1 — Section header synonym map
│   ├── enricher.py        # Phase 2 — Named entity recognition
│   ├── entity_types.py    # Phase 2 — Entity taxonomy definitions
│   ├── feature_vector.py  # Phase 2 — 40D vector construction
│   ├── semantic_map.py    # Phase 2 — Entity → type mapping
│   ├── classifier.py      # Phase 3 — Model inference wrapper
│   ├── model_io.py        # Phase 3 — Artifact load/save
│   ├── dataset.py         # Phase 3 — Training data loader
│   ├── trainer.py         # Phase 3 — Training orchestration
│   ├── scorer.py          # Phase 4 — Hybrid scoring engine
│   ├── gap_analyzer.py    # Phase 5 — Skills gap computation
│   ├── visualizer.py      # Phase 5 — Radar chart generation
│   ├── report_generator.py# Phase 5 — PDF report creation
│   └── app.py             # Phase 6 — Flask application factory
├── static/
│   ├── css/style.css      # Main UI styles (futuristic dark design)
│   ├── css/intro.css      # Cinematic intro animation overlay
│   └── js/
│       ├── app.js         # Frontend logic (form submission, results rendering)
│       └── intro.js       # Intro animation controller
├── templates/
│   └── index.html         # Single-page application template
├── config/
│   ├── training_config.yaml  # ML hyperparameters
│   └── scoring_config.yaml   # Intent affinity map + penalty rules
├── data/
│   └── niche_benchmarks.json # Human-readable skill names per niche
├── models/
│   └── 20260423_155216_seed42.joblib  # Final trained artifact
└── run.py                 # Entry point (Waitress WSGI)
```

#### Phase 3 — ML Inference and Prediction Logging

The `predict_niche()` function in `src/classifier.py` wraps the scikit-learn model's `predict_proba()` call and adds two important production behaviours:

1. **Probability-based prediction:** Rather than calling `model.predict()` (which returns a single class), the system calls `model.predict_proba()` and reads the full probability vector across all 6 niches. The class with the highest probability is the predicted niche, and that probability value is returned as the `confidence` score (range: 0.0–1.0). This allows the frontend to display how certain the model is, and also feeds the hybrid scoring layer which uses all 6 probabilities.

2. **Privacy-preserving prediction logging:** Every prediction call — success or failure — appends one JSON line to `logs/predictions.jsonl`. The log entry includes timestamp, model version, prediction result, confidence score, and an 8-character SHA-256 hash of the feature vector. The raw feature vector is never logged. The hash is computed as:
   ```
   SHA-256( str(sorted(enumerate(feature_vector))) )[:8]
   ```
   This lets operators correlate two calls on the same CV (same hash) without being able to reconstruct the original vector — satisfying the PII protection requirement.

---

### 5.2 Model Training

**Training procedure:**
1. Load 2,400 labeled CV feature vectors from `data/training/dataset.csv`
2. Split 80/20 train/test (1,920 training samples, 480 test samples)
3. Fit `RandomForestClassifier` with the hyperparameters in Section 4.2
4. After fitting, compute per-niche centroid vectors (mean feature vector per class) and attach to the model artifact — these are required by Phase 4's cosine similarity scoring
5. Evaluate on the held-out 20% test set
6. Persist the full artifact (classifier + centroids + metadata) as a `.joblib` file
7. Write a JSON experiment record with all metrics, hyperparameters, and dataset hash

**Training time:** Under 60 seconds on a standard laptop CPU (Python 3.11, Windows 11).

**Artifact size:** Approximately 15–20 MB for 200 trees on a 40-feature dataset.

**Reproducibility:** Fixed `random_state=42`. The same dataset + same hyperparameters produce a bitwise identical model artifact every run. The dataset hash is stored in the experiment record to detect any accidental dataset mutation.

### 5.3 Model Tuning

Three training runs were performed with incremental improvements:

| Run ID | Date | Dataset Size | Accuracy | Key Change |
|--------|------|-------------|----------|-----------|
| `20260418_005030_seed42` | 2026-04-18 | Initial | — | Baseline training |
| `20260423_133324_seed42` | 2026-04-23 | Expanded | — | Dataset expansion |
| `20260423_155216_seed42` | 2026-04-23 | 2,400 | **85.625%** | Final tuned version |

**Tuning approach:** Manual hyperparameter selection based on domain knowledge rather than automated grid search, for the following reasons:
- The feature space is small (40 dimensions) and well-understood
- The training set is balanced by design, so `class_weight="balanced"` is a deterministic choice
- The primary tuning lever was `n_estimators` (tested 100, 150, 200) — 200 was selected for prediction stability
- `min_samples_leaf=2` was added to address minor overfitting observed in the 100-tree baseline (training accuracy was near 100% but test accuracy was lower)

### 5.4 Skills Gap Analysis Engine (Phase 5)

Phase 5 consists of three components that together convert the scoring result into actionable output.

**Gap Analyzer (`src/gap_analyzer.py`)**

The gap analyzer answers: *"Which specific skills is this student missing for their top recommended niche?"* It works by:

1. Loading `data/niche_benchmarks.json` — a curated list of the top 10 most important skill dimensions for each niche, in importance order
2. Retrieving the `niche_top_skills` from the trained model artifact — the 10 feature vector dimensions with the highest centroid values for the target niche, sorted by importance descending
3. Mapping each of those 10 dimensions back to its human-readable benchmark name (e.g., dimension index 24 → "ROS / ROS2")
4. For each of the 10 benchmark skills, checking whether the corresponding dimension in the student's feature vector is > 0 (present) or = 0 (missing)

The positional alignment between `niche_top_skills` (model artifact) and `top_skills` (benchmark JSON) is a fixed contract: rank-0 in both lists refers to the same importance rank. This means the gap output is always sorted by skill importance, not alphabetically.

**Radar Chart Generator (`src/visualizer.py`)**

The radar chart is a 6-spoke polar plot rendered using Matplotlib's Agg (non-interactive) backend. It is mandatory to set `matplotlib.use("Agg")` before any pyplot import because the server runs headless (no display). The chart:
- Plots `composite_score` (0.0–1.0) on each of the 6 spokes in `CANONICAL_NICHES` order
- Uses a dark background (`#0B1120`) with sky-blue fill and line, and slate-colored grid to match the application's visual design
- Is generated entirely in RAM as PNG bytes via `io.BytesIO` — never written to disk
- Is returned base64-encoded for embedding directly in the HTML response

**PDF Report Generator (`src/report_generator.py`)**

The PDF is assembled using ReportLab's PLATYPUS (Page Layout and Typography Using Scripts) framework entirely in memory. It contains:
1. A ranked table of the student's top 3 niche recommendations with composite fit scores
2. A skills gap section for the top niche listing present vs. missing skills
3. The radar chart embedded as a PNG image
4. A generation timestamp

The PDF bytes are returned via the API as a downloadable attachment — no file is ever written to disk, satisfying the privacy-preserving in-memory constraint.

---

## 6. RESULTS AND EVALUATION

### 6.1 Metrics

**Final model: `20260423_155216_seed42`**

**Overall Accuracy: 85.625%** (on 480 held-out test samples)

**Per-Niche Performance (test set, 80 samples per niche):**

| Niche | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Embedded Systems | 0.946 | 0.875 | 0.909 | 80 |
| Automotive | 0.909 | 0.875 | 0.892 | 80 |
| Mechanical Design | 1.000 | 0.838 | 0.912 | 80 |
| Robotics | 0.917 | 0.825 | 0.868 | 80 |
| Industrial Automation | 0.868 | 0.738 | 0.797 | 80 |
| Technical Management | 0.648 | 0.988 | 0.782 | 80 |

**Observations:**
- **Mechanical Design** achieves perfect precision (1.000) — every student the model predicts as Mechanical Design actually belongs there.
- **Technical Management** has the lowest F1 (0.782) but very high recall (0.988) — the model rarely misses a management-oriented student, but does occasionally assign management to non-management students. This is the hardest niche to distinguish because management skills appear as secondary skills across all other niches.
- **All niches exceed the minimum F1 threshold of 0.60**, confirming the model does not systematically fail any single engineering specialisation.
- Overall accuracy of **85.625% exceeds the 80% target** set in the project objectives.

### 6.2 Top Feature Importances

The Random Forest's built-in feature importance scores reveal which skill dimensions drive classification decisions most:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Robotic framework — Skills section | 0.0971 |
| 2 | Robotic framework — Projects section | 0.0808 |
| 3 | Microcontroller — Skills section | 0.0768 |
| 4 | Microcontroller — Projects section | 0.0647 |
| 5 | Mechanical tool — Projects section | 0.0619 |
| 6 | Automotive standard — Skills section | 0.0585 |
| 7 | Automotive standard — Projects section | 0.0571 |
| 8 | PLC hardware — Skills section | 0.0551 |
| 9 | PLC hardware — Projects section | 0.0539 |
| 10 | CAD software — Projects section | 0.0508 |

**Key insight:** The model weights *projects-section* mentions of skills almost as heavily as *skills-section* mentions. This validates the design decision to track entity location (section) as a separate encoding dimension — a student who has only listed a tool under Education carries a weaker signal than one who has built projects with it.

### 6.3 Analysis

- The model correctly handles the most common confusion cases: Embedded Systems vs. Automotive (both use microcontrollers and communication protocols), and Robotics vs. Industrial Automation (both use PLCs and sensors). The section-aware feature encoding is the primary factor that separates these pairs.
- Technical Management is the most commonly confused niche because management keywords (Agile, Scrum, MS Project) appear as secondary skills on many non-management CVs. The model compensates with high recall at the cost of lower precision.
- The hybrid scoring layer (Phase 4) further improves user-facing results by incorporating preference signals, which means even when the raw classifier is uncertain between two similar niches, the student's stated preferences break the tie correctly.

### 6.4 Visualizations (Generated at Runtime)

The system generates the following visualizations for each analysis:

1. **Radar Chart** — 6-axis plot showing the student's skill coverage across all entity types, overlaid against the top recommended niche's benchmark profile. Generated with Matplotlib on a dark background (#0B1120) with sky-blue line and slate grid.
2. **Niche Score Bar Chart** — Horizontal bar visualization of all 6 composite scores, color-coded by rank.
3. **Skills Gap Progress Bars** — Per-skill progress indicators for the top 3 recommended niches, showing present vs. missing skills against the benchmark.

---

## 7. DEPLOYMENT

The system is deployed as a web application with the following architecture:

**Backend:**
- Python 3.11 / Flask 3.x application factory pattern
- Waitress WSGI server (production-grade, replaces Flask's dev server)
- Single worker with 4 threads to handle concurrent requests
- 180-second timeout per request (PDF parsing on large files can take time)
- HTTP security headers: CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy

**Frontend:**
- Single-page application (no JavaScript framework — vanilla JS + CSS)
- Three-stage page flow: cinematic intro animation → landing page → analysis form → results
- Intro animation: 8.3-second CSS-only cinematic sequence (hex grid, orbital ring, particle drift, letter-by-letter wordmark reveal) with sessionStorage gate (plays once per session)
- Drag-and-drop PDF upload with animated progress states
- Results rendered dynamically from JSON API response

**API:**
- `POST /api/v1/run` — accepts multipart PDF upload + optional preference parameters; returns JSON with ranked niches, gap analysis, radar chart (base64 PNG), and downloadable PDF report URL
- `GET /api/v1/health` — returns server status and model_loaded flag

**Infrastructure:**
- Configured for deployment on Render (render.yaml included)
- PORT environment variable supported for cloud hosting
- Git repository: public on GitHub at `github.com/Mohamad-Shabrawy/Mechatronics-Career-Architect`

**Capacity:** Waitress with 4 threads can handle approximately 4 simultaneous long-running analyses. For a student demo or classroom setting this is sufficient. Scaling to more concurrent users would require adding workers or a task queue (Celery + Redis).

---

## 8. CONCLUSION AND FUTURE WORK

### 8.1 Conclusion

This project successfully delivered an end-to-end AI-powered career guidance system for Mechatronics engineering students. The key findings are:

1. A 40-dimensional skill-section feature encoding extracted from PDF CVs is sufficient to classify engineering niches with **85.625% accuracy** using a Random Forest classifier — significantly above the 80% target.
2. The hybrid scoring formula (60% CV signal + 40% preference intent) produces more personally relevant rankings than the raw classifier alone, particularly for students who have overlapping skill sets across adjacent niches.
3. The skills gap analysis gives students specific, actionable feedback — not just "you should consider robotics" but "you are missing ROS2, Gazebo, and motion planning frameworks compared to the Robotics benchmark profile."
4. The full pipeline from PDF upload to ranked results runs in under 3 minutes on typical student CVs, making it practical for real use.
5. The system is privacy-preserving: no CV data is persisted to disk at any point.

### 8.2 Future Improvements

1. **Real CV dataset:** Replace the synthetic training data with a real labeled dataset of anonymized student CVs, which would improve generalization and surface niche boundaries that synthetic data cannot capture.
2. **Transfer learning on CV text:** Replace the keyword-counting feature extraction with a pre-trained language model (e.g., BERT fine-tuned on engineering job descriptions) to capture semantic meaning beyond exact keyword matches.
3. **Arabic CV support:** Add Arabic-language keyword lexicons and right-to-left PDF layout parsing to serve Arabic-medium students at Egyptian universities.
4. **Niche trajectory modeling:** Track a student's CV over time (semester by semester) and model their progression toward a target niche, not just their current snapshot.
5. **Industry validation:** Partner with Mechatronics employers to validate and update the niche benchmark profiles annually as the technology landscape evolves.
6. **Expanded niche taxonomy:** Add sub-niches (e.g., distinguishing ADAS/autonomous vehicles within Automotive, or medical robotics within Robotics) as the model is trained on more data.
7. **Explainability layer:** Add SHAP (SHapley Additive exPlanations) values to the API response so the UI can show the student exactly which CV keywords drove each niche score, not just the final number.

---

## 9. REFERENCES

> **Note to report writer:** Format all references in IEEE style. Sources below include the confirmed data sources used in this project plus the primary technical references.

**ML & Algorithms:**
1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill. (Cosine similarity basis)

**Data Sources:**
4. Wuzzuf.com. (2024). *Job listings dataset — Egypt engineering sector*. Scraped from Wuzzuf.com, Egypt's largest job platform.
5. Kaggle. (2024). *Engineering job descriptions dataset* (`job_dataset.csv`). Retrieved from https://www.kaggle.com/
6. Kaggle. (2024). *AI-based Career Recommendation System dataset* (`AIbased_Career_Recommendation_System.csv`). Retrieved from https://www.kaggle.com/

**Libraries & Frameworks:**
7. PyMuPDF Documentation. (2024). *PyMuPDF — Python binding for MuPDF*. Retrieved from https://pymupdf.readthedocs.io/
8. Flask Documentation. (2024). *Flask — A lightweight WSGI web application framework*. Retrieved from https://flask.palletsprojects.com/
9. ReportLab Inc. (2024). *ReportLab PDF Library*. Retrieved from https://www.reportlab.com/
10. Waitress Documentation. (2024). *Waitress — Pure-Python WSGI Server*. Retrieved from https://docs.pylonsproject.org/projects/waitress/
11. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.

---

## 10. APPENDIX

### A. Feature Vector Index Map (all 40 dimensions)

| Index | Entity Type | CV Section |
|-------|-------------|------------|
| 0 | PLC Hardware | Skills |
| 1 | PLC Hardware | Projects |
| 2 | PLC Hardware | Education |
| 3 | PLC Hardware | Internships |
| 4 | CAD Software | Skills |
| 5 | CAD Software | Projects |
| 6 | CAD Software | Education |
| 7 | CAD Software | Internships |
| 8 | Microcontroller | Skills |
| 9 | Microcontroller | Projects |
| 10 | Microcontroller | Education |
| 11 | Microcontroller | Internships |
| 12 | Communication Protocol | Skills |
| 13 | Communication Protocol | Projects |
| 14 | Communication Protocol | Education |
| 15 | Communication Protocol | Internships |
| 16 | Programming Language | Skills |
| 17 | Programming Language | Projects |
| 18 | Programming Language | Education |
| 19 | Programming Language | Internships |
| 20 | Simulation Tool | Skills |
| 21 | Simulation Tool | Projects |
| 22 | Simulation Tool | Education |
| 23 | Simulation Tool | Internships |
| 24 | Robotic Framework | Skills |
| 25 | Robotic Framework | Projects |
| 26 | Robotic Framework | Education |
| 27 | Robotic Framework | Internships |
| 28 | Automotive Standard | Skills |
| 29 | Automotive Standard | Projects |
| 30 | Automotive Standard | Education |
| 31 | Automotive Standard | Internships |
| 32 | Mechanical Tool | Skills |
| 33 | Mechanical Tool | Projects |
| 34 | Mechanical Tool | Education |
| 35 | Mechanical Tool | Internships |
| 36 | Management Methodology | Skills |
| 37 | Management Methodology | Projects |
| 38 | Management Methodology | Education |
| 39 | Management Methodology | Internships |

### B. Hybrid Scoring Formula (Full Detail)

```
For each niche n ∈ {robotics, embedded_systems, automotive,
                    industrial_automation, mechanical_design, technical_management}:

  1. CV Component:
     centroid_n = mean feature vector of all training samples labeled n
     cosine_n   = dot(fv, centroid_n) / (||fv|| × ||centroid_n||)
     (returns 0.0 if fv is the zero vector)

  2. Intent Component:
     base = 0.5
     for each preference p expressed by the user:
         delta = affinity_map[p][value][n]   (from scoring_config.yaml)
         base += delta
     intent_n = clamp(base, 0.0, 1.0)

  3. Composite:
     composite_n = 0.60 × cosine_n + 0.40 × intent_n

  4. Override Penalty:
     penalized_n = composite_n − penalty_n   (penalty_n ≥ 0.0)

  5. Normalization:
     max_score     = max(penalized_n for all n)
     normalized_n  = penalized_n / max_score   (top niche = 1.0)
```

### C. Technology Stack Summary

```
Core ML Pipeline:      scikit-learn, NumPy, Pandas, joblib
PDF Processing:        PyMuPDF (fitz)
Web Application:       Flask, Waitress
Visualization:         Matplotlib
Report Generation:     ReportLab
Configuration:         PyYAML
Testing:               pytest (495 tests, 0 regressions)
Frontend:              Vanilla HTML/CSS/JavaScript
Version Control:       Git / GitHub
```

### D. Model Artifact Metadata

```
Run ID:              20260423_155216_seed42
Training Date:       2026-04-23T15:52:17Z
Algorithm:           RandomForestClassifier
Dataset Size:        2,400 samples (400 per niche)
Train/Test Split:    80% / 20%
Overall Accuracy:    85.625%
Min Niche F1:        0.782 (Technical Management)
Max Niche F1:        0.912 (Mechanical Design)
Random Seed:         42
Dataset Hash:        da8633f09af452bef50d8cf4d457c5180081b455ebae7b5d8d5cdb24c4a1e277
```

---

## INSTRUCTIONS FOR CLAUDE CHAT (REPORT WRITER)

When generating the Word report from this brief:

1. **Format:** Academic report format. Title page, table of contents, numbered sections as structured above.
2. **Language:** Formal English throughout. No casual phrasing.
3. **Depth:** Expand each section with full paragraphs. The numbers and technical facts in this brief are exact — do not change them. For narrative/context (e.g., "why mechatronics is important"), you may enrich with relevant background.
4. **Figures:** Where the brief mentions a visualization (radar chart, bar chart, etc.), note in the report that the figure is available as output of the system and describe what it shows.
5. **Dataset section (3.1–3.2):** The student will provide additional details about real-world data sources. Leave a clearly marked `[STUDENT TO ADD: ...]` placeholder for anything the brief marks as "user will provide."
6. **References:** Format all references in IEEE style. Add the `[STUDENT TO ADD: additional references]` placeholder at the end.
7. **Page estimate:** Target 15–25 pages for a complete academic submission.
