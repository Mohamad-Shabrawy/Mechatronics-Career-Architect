# Mechatronics Career Architect — Claude Code Instructions

## Project

Flask web app: AI-powered career niche analysis for mechatronics engineering students.
Code: `c:/Python Engineering Projects/mechatronics-career-architect/`
Run locally: `python run.py` → http://127.0.0.1:5002

**Python version: 3.11+** — do not use syntax or APIs unavailable in 3.11.

**In-memory only** — all CV processing, chart rendering, and PDF generation happen entirely
in RAM. No pipeline result is ever written to disk. This is a hard privacy constraint; do not
introduce any file writes inside the pipeline (phases 1–5) or the API layer.

All 6 phases complete. Never modify pipeline logic or ML model without explicit instruction.

---

## Frontend Design Standard

This project uses the **Antigravity design system** (see `.claude/skills/FRONTEND_DESIGN.md`).
Apply it automatically for ALL UI work — components, pages, layouts, animations.

### Before writing any UI code

Commit to a bold aesthetic direction:
- **Purpose**: What problem does this interface solve? Who uses it?
- **Tone**: Pick an extreme and execute it with precision
- **Differentiation**: What is the one thing someone will remember about this?

Intentionality is the standard — bold maximalism and refined minimalism both work. What fails is being generic.

### Typography

- NEVER use: Inter, Roboto, Arial, Space Grotesk, system-ui, or any system font default
- Always pair a distinctive display font with a refined body font
- Typography must feel genuinely designed for the context, not picked from a shortlist

### Color

- Commit to a cohesive aesthetic using CSS variables
- Dominant colors with sharp accents outperform evenly-distributed timid palettes
- NEVER default to purple gradients on white backgrounds

### Animation

- Prioritise CSS-only solutions
- One well-orchestrated page load with staggered reveals beats scattered micro-interactions
- High-impact moments: hover states and reveals that genuinely surprise

### Hard rules — never do these

- No generic font families (see Typography above)
- No clichéd color schemes (purple on white is the canonical example)
- No predictable cookie-cutter component patterns
- Every design must have a clear, committed point-of-view

---

## UI Skill Tools

- `ui-ux-pro-max` skill — use for specific palette, font pairing, style, and stack lookups
- `FRONTEND_DESIGN.md` skill — the philosophy and hard rules above

When doing UI work, run the design system query first, then apply the hard rules from this file.

---

## Backend Rules

- Never change pipeline logic (phases 1–5) without explicit instruction
- Never retrain or replace the model artifact without explicit instruction
- Port: 5002 locally. Ports 5000 and 5001 have permanent zombie sockets — never use them.
- Tests: `python -m pytest tests/ -q` — must stay at 495 passing, zero regressions
