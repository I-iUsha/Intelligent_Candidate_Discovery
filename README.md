# Redrob Intelligent Candidate Ranker

**Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**

A multi-pillar hybrid scoring system that ranks candidates the way a great recruiter would — not by matching keywords, but by understanding who genuinely fits the role.

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker

# 2. Install dependencies (no GPU required)
pip install -r requirements.txt

# 3. Place the candidates file in the repo root
cp /path/to/candidates.jsonl ./candidates.jsonl

# 4. Run the ranker (produces submission CSV)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 5. Validate before submitting
python validate_submission.py submission.csv
```

**Single reproduce command:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## What This System Does Differently

Most candidate rankers do one thing: embed a JD and embed candidate profiles, then rank by cosine similarity. That approach has three well-known failure modes:

1. **Keyword stuffers win** — a candidate who lists every buzzword scores high regardless of whether they've actually used them
2. **Unavailable candidates rank first** — a perfect-on-paper candidate who hasn't responded to recruiters in 6 months is useless
3. **Honeypots pass through** — impossible profiles (8 years at a company founded 3 years ago) fool embedding models

This system addresses all three explicitly.

---

## Architecture

### 5-Stage Pipeline

```
100K candidates
      │
      ▼
┌─────────────────────┐
│  Stage 1: Honeypot  │  Detect impossible profiles → score 0.0
│  Detection          │  (YoE contradictions, future dates, expert
└─────────────────────┘  skills with 0 endorsements)
      │
      ▼
┌─────────────────────┐
│  Stage 2: Hard      │  Non-engineering titles (Marketing Manager,
│  Title Filter       │  Accountant, etc.) → title_corroboration = 0.15
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Stage 3: 4-Pillar  │  Technical Score + Contextual Score
│  Scoring            │  + Availability Score + Behavioral Signals
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Stage 4: Weighted  │  55% Technical × 25% Contextual × 20% Availability
│  Combination        │  × Consulting penalty (if applicable)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Stage 5: Reasoning │  Per-candidate reasoning from actual profile
│  Generation         │  facts (no hallucination)
└─────────────────────┘
      │
      ▼
   Top 100 CSV
```

---

## Scoring Pillars

### Pillar A — Skill Fit (Cluster-Weighted Quality)

Skills are grouped into 8 semantic clusters aligned to the JD:

| Cluster | JD Weight | Examples |
|---|---|---|
| `core_retrieval` | 1.00 | FAISS, Qdrant, Milvus, Elasticsearch, BM25, vector search |
| `embedding_systems` | 1.00 | Sentence Transformers, BGE, E5, OpenAI embeddings |
| `ml_ranking` | 0.90 | Learning to Rank, NDCG, MRR, A/B testing, recsys |
| `llm_rag` | 0.85 | RAG, LangChain, LlamaIndex, HuggingFace, prompt engineering |
| `python_engineering` | 0.75 | Python, FastAPI, Pandas, async |
| `fine_tuning` | 0.70 | LoRA, QLoRA, PEFT, instruction tuning |
| `ml_infrastructure` | 0.65 | MLOps, BentoML, ONNX, MLflow |
| `adjacent_nlp` | 0.60 | NLP, text classification, NER, summarization |

Each skill is scored on **quality** (not just presence):
- 40% proficiency level (expert/advanced/intermediate/beginner)
- 35% duration in months (log-scaled, 60mo = max)
- 25% endorsements received (log-scaled, 50 = max)

This is the anti-keyword-stuffer mechanism: a "beginner" skill listed with 0 endorsements and 2 months of use contributes almost nothing even if the keyword matches.

### Pillar B — Career Depth in AI/ML

Scans all career history descriptions for AI/ML engineering evidence. Each role is weighted by how deeply AI-related it is (based on keyword density in the description + title alignment) × its duration in months.

Result: a 0–1 score representing what fraction of the candidate's actual working time was in genuine AI/ML engineering. A candidate who lists "RAG" as a skill but whose career descriptions mention only "stakeholder management" and "Excel" scores near 0.

### Pillar C — Title & Career Corroboration

Cross-checks whether listed skills appear as evidence in career descriptions. A Marketing Manager with 15 AI skills listed but zero engineering career text scores 0.15. A Senior ML Engineer with corroborated skills scores 1.0.

### Pillar D — Experience Fit

Soft scoring against the JD's 5–9 year range:
- 6–8 years → 1.00 (sweet spot)
- 5–9 years → 0.90
- 4 years → 0.75 (junior but possible)
- 9–12 years → 0.75 (senior but slightly overqualified)
- <3 years → 0.30

---

## Behavioral Availability Multiplier

Availability is a **multiplier** on the combined score, not just another pillar. A perfect-on-paper candidate who is behaviorally unavailable gets downweighted significantly.

Availability is composed of:
- **35%** Recency of last login (inactive 6+ months → 0.20)
- **25%** Recruiter response rate (<20% → 0.40)
- **25%** Notice period (≤30 days → 1.0; >120 days → 0.40)
- **15%** Interview completion rate

Open-to-work flag applies a ×1.1 boost; not open applies ×0.85.

---

## Honeypot Detection

The dataset contains ~80 honeypot candidates with subtly impossible profiles. Four checks are applied:

1. **YoE contradiction** — stated years > 2.5× actual career history length
2. **Impossible skill breadth** — 5+ "expert" skills with 0 endorsements and <6 months duration
3. **Future dates** — career start dates in the future
4. **Suspicious tenure** — 10+ years at a 1–10 person company
5. **Impossible breadth for YoE** — 15+ advanced/expert skills with <3 years experience

Honeypots are scored 0.0 and excluded from the ranked list entirely.

---

## Location Scoring

The JD specifies Pune/Noida-preferred, with Hyderabad, Mumbai, Delhi NCR, and Bangalore also welcome. International candidates outside India are significantly downweighted unless willing to relocate.

| Location | Score |
|---|---|
| Pune or Noida | 1.00 |
| Bangalore, Hyderabad, Mumbai, Delhi NCR | 0.90 |
| Other India city | 0.75–0.82 |
| India, willing to relocate | 0.82 |
| Outside India, willing to relocate | 0.55 |
| Outside India, not willing | 0.30 |

---

## Consulting-Only Career Penalty

The JD explicitly states candidates whose entire career is at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc. are a poor fit. Candidates with >85% of career tenure at these firms receive a ×0.70 multiplier on their final score. Candidates who have some consulting history but also product-company experience are not penalized.

---

## Final Score Formula

```
technical_score = (
    0.35 × skill_fit +
    0.30 × ai_career_depth +
    0.15 × title_corroboration +
    0.10 × education +
    0.10 × github_activity
)

contextual_score = (
    0.50 × experience_fit +
    0.30 × location_fit +
    0.20 × platform_engagement
)

final_score = (
    0.55 × technical_score +
    0.25 × contextual_score +
    0.20 × availability
) × consulting_penalty
```

---

## Compute Constraints

| Constraint | Limit | This System |
|---|---|---|
| Runtime | ≤ 5 minutes | ~63 seconds |
| Memory | ≤ 16 GB | ~2 GB peak |
| GPU | Not allowed | CPU only ✅ |
| Network | Not allowed | Zero external calls ✅ |

No embeddings model is required at runtime. All scoring is rule-based and heuristic, operating directly on structured candidate fields. This makes the system fast, deterministic, and fully reproducible.

---

## File Structure

```
redrob-ranker/
├── rank.py                    # Main ranker (alias for ranker.py)
├── ranker.py                  # Full source — scoring engine
├── requirements.txt           # Python dependencies
├── submission_metadata.yaml   # Hackathon metadata
├── validate_submission.py     # Official format validator (from bundle)
└── README.md                  # This file
```

---

## Reproducing the Submission

```bash
# Full reproduce command (≤ 5 minutes, CPU only, no network)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Then validate
python validate_submission.py submission.csv
# Expected: "Submission is valid."
```

The ranker is fully deterministic — same input always produces the same output.

---

## Design Decisions & Tradeoffs

**Why not use sentence-transformers for embedding?**
The 5-minute CPU constraint with 100K candidates makes per-candidate embedding impractical with a full transformer model. A 384-dim model embedding 100K candidates takes 8–15 minutes on CPU. The cluster-based skill scoring achieves similar semantic coverage (NLP ≈ natural language processing; vector search ≈ dense retrieval) without the compute cost.

**Why is availability a multiplier rather than a pillar?**
Because a candidate who scores 0.95 on skills but has a 5% response rate and hasn't logged in for 8 months is, for practical hiring purposes, not actually available. Making it a multiplier means it can't be "averaged away" by strong technical scores.

**Why are skills quality-weighted rather than presence-weighted?**
A keyword stuffer lists 30 AI skills at "beginner" proficiency with 0 endorsements. A genuine engineer has 8 skills, 3 of which are "advanced" with 20+ endorsements and 24+ months of use. The quality scoring (proficiency × duration × endorsements) separates these clearly.

**Why not fine-tune a learning-to-rank model?**
No labeled training data is provided. A learned ranker without training labels would just overfit to proxy signals. The explicit rule-based system encodes the JD's stated priorities directly and is interpretable at Stage 4 review.
