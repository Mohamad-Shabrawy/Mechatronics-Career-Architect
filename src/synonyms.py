"""
synonyms.py — Section Header Recognition Dictionary

Real CVs don't all say "Skills" and "Internships" in exactly those words.
One person writes "Technical Skills", another writes "Work Experience",
another writes "Professional Background". This file maps all those variants
to the four canonical section names our system uses.

How it works:
- The keys are the four section names our output schema always uses.
- The values are lists of lowercase strings that are recognized as headers
  for that section in a CV.
- Matching is EXACT (after lowercasing and stripping trailing colons/spaces).
  We deliberately avoid substring matching here because a line like
  "experience working with PLC systems for 3 years" starts with "experience"
  but is clearly body text, not a header. The length guard in the parser
  (lines > 60 chars are rejected) is the first filter; exact match is the second.

The synonym lists are sorted roughly by how common they are on real CVs.
Feel free to expand them in future phases — just keep them lowercase.
"""

SECTION_SYNONYMS: dict = {

    # ── SKILLS ────────────────────────────────────────────────────
    # This section captures technical and professional competencies.
    # It's the most variably-named section on engineering CVs.
    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "key skills",
        "professional skills",
        "competencies",
        "technical competencies",
        "tools & technologies",
        "tools and technologies",
        "tools",
        "technologies",
        "software skills",
        "hard skills",
        "areas of expertise",
        "expertise",
        "technical expertise",
        "skill set",
        "skillset",
        "software & tools",
        "software and tools",
        "programming skills",
        "engineering skills",
        "laboratory skills",
        "lab skills",
    ],

    # ── PROJECTS ──────────────────────────────────────────────────
    # Covers academic, personal, and professional project work.
    # Mechatronics students tend to have strong project sections —
    # this is where the most technical keywords usually live.
    "projects": [
        "projects",
        "academic projects",
        "personal projects",
        "key projects",
        "project experience",
        "notable projects",
        "selected projects",
        "relevant projects",
        "engineering projects",
        "project work",
        "capstone project",
        "graduation project",
        "final year project",
        "design projects",
        "research projects",
        "hands-on projects",
    ],

    # ── EDUCATION ─────────────────────────────────────────────────
    # University degrees, certifications, online courses, training programs.
    # Sometimes blended with "Certifications" in a single section.
    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "qualifications",
        "educational background",
        "academic history",
        "degrees",
        "certifications",
        "training",
        "courses",
        "education & certifications",
        "education and certifications",
        "academic credentials",
        "academic experience",
        "academic profile",
        "university",           # some CVs use this as a standalone header
        "certificates",
        "professional development",
        "continuing education",
    ],

    # ── INTERNSHIPS ───────────────────────────────────────────────
    # Work experience in any form — internships, co-ops, full-time roles.
    # "Experience" and "Work Experience" are the most common real-world headers.
    # "Internship" (singular) appears on student CVs.
    "internships": [
        "internships",
        "internship",
        "work experience",
        "experience",
        "professional experience",
        "employment history",
        "employment",
        "career history",
        "work history",
        "industrial training",
        "industry experience",
        "practical experience",
        "on-the-job experience",
        "career experience",
        "job experience",
        "field experience",
        "training experience",
        "summer training",      # very common on Egyptian engineering CVs
        "industrial internship",
        "cooperative education",
        "co-op",
    ],
}
