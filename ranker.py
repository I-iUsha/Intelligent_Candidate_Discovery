"""
Redrob Intelligent Candidate Ranker
====================================
Design philosophy:
  - Multi-pillar hybrid scoring (not pure keyword matching)
  - Hard disqualifiers first (honeypots, obviously irrelevant titles)
  - 4 scoring pillars: Semantic Fit, Career Depth, Skill Trust, Behavioral Availability
  - Anti-keyword-stuffer: title/career history must corroborate listed skills
  - Runs in < 5 minutes on CPU, 16GB RAM, no network

Architecture:
  1. Fast pre-filter: remove obviously irrelevant candidates (non-tech titles, wrong country signals)
  2. Feature extraction: structured features from each candidate
  3. Pillar scoring: 4 independent scoring pillars
  4. Weighted combination with behavioral multiplier
  5. Honeypot detection: flag candidates with impossible profiles
  6. Final ranking + reasoning generation
"""

import json
import math
import re
import argparse
from datetime import datetime, date
from collections import defaultdict
from typing import Optional

import numpy as np

# ─── JOB DESCRIPTION (parsed manually for precision) ───────────────────────────

JD = {
    "role": "Senior AI Engineer",
    "company_stage": "Series A startup",

    # HARD REQUIREMENTS (must-haves)
    "must_have_skills": [
        "embeddings", "sentence-transformers", "vector search", "semantic search",
        "information retrieval", "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
        "OpenSearch", "Elasticsearch", "RAG", "retrieval", "ranking",
        "Python", "NLP", "transformer", "LLM", "language model",
        "evaluation framework", "NDCG", "MRR", "MAP", "A/B testing",
        "hybrid search", "dense retrieval", "BM25",
    ],

    # NICE TO HAVE
    "nice_to_have_skills": [
        "LoRA", "QLoRA", "PEFT", "fine-tuning", "learning to rank",
        "XGBoost", "recommendation system", "search engine",
        "distributed systems", "inference optimization", "open source",
        "recsys", "LangChain", "LlamaIndex", "vector database",
        "PyTorch", "TensorFlow", "HuggingFace", "transformers",
        "Prompt Engineering", "BGE", "E5", "sentence transformers",
        "OpenAI", "embeddings API", "re-ranking",
    ],

    # DISQUALIFIER SIGNALS from the JD text
    "disqualifier_titles": [
        "marketing", "sales", "accountant", "finance", "hr ", "human resources",
        "recruiter", "operations manager", "customer support", "civil engineer",
        "mechanical engineer", "electrical engineer", "graphic designer",
        "content writer", "seo", "business development", "project coordinator",
        "supply chain", "logistics",
    ],

    # JD explicitly mentions these negative signals
    "disqualifier_company_types": [
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "mphasis", "hexaware", "l&t infotech",
    ],

    # Preferred locations
    "preferred_locations": ["pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
                             "bengaluru", "india"],

    "target_yoe_min": 5,
    "target_yoe_max": 9,
    "preferred_yoe_min": 6,
    "preferred_yoe_max": 8,

    # Behavioral preferences from JD
    "preferred_notice_max": 30,
    "acceptable_notice_max": 60,
}

# Extended skill taxonomy — semantic clusters for skill matching
SKILL_CLUSTERS = {
    "core_retrieval": [
        "faiss", "milvus", "pinecone", "weaviate", "qdrant", "opensearch",
        "elasticsearch", "vector search", "semantic search", "information retrieval",
        "dense retrieval", "sparse retrieval", "hybrid search", "bm25", "ann",
        "approximate nearest neighbor", "vector database", "vector db",
    ],
    "embedding_systems": [
        "embeddings", "sentence-transformers", "sentence transformers",
        "openai embeddings", "bge", "e5", "embedding model", "text embeddings",
        "word2vec", "doc2vec", "bi-encoder", "cross-encoder",
    ],
    "llm_rag": [
        "rag", "retrieval augmented generation", "llm", "large language model",
        "gpt", "chatgpt", "openai", "anthropic", "langchain", "llamaindex",
        "llama index", "prompt engineering", "in-context learning",
        "huggingface", "transformers", "bert", "roberta",
    ],
    "ml_ranking": [
        "learning to rank", "ranknet", "lambdamart", "xgboost ranking",
        "ranking system", "recommendation system", "recsys", "collaborative filtering",
        "ndcg", "mrr", "map", "precision@k", "offline evaluation", "a/b testing",
    ],
    "fine_tuning": [
        "fine-tuning", "fine tuning", "lora", "qlora", "peft", "instruction tuning",
        "rlhf", "dpo", "sft", "parameter efficient", "adapter",
    ],
    "ml_infrastructure": [
        "mlops", "model serving", "inference optimization", "bentoml", "triton",
        "onnx", "quantization", "distributed training", "pytorch", "tensorflow",
        "model deployment", "ml pipeline", "kubeflow", "mlflow",
    ],
    "python_engineering": [
        "python", "fastapi", "flask", "django", "asyncio", "pydantic",
        "sqlalchemy", "pandas", "numpy", "scipy",
    ],
    "adjacent_nlp": [
        "nlp", "natural language processing", "text classification", "ner",
        "named entity recognition", "sentiment analysis", "summarization",
        "question answering", "text mining", "tokenization",
    ],
}

# Weight each cluster by JD importance
CLUSTER_WEIGHTS = {
    "core_retrieval": 1.0,     # Must-have
    "embedding_systems": 1.0,  # Must-have
    "llm_rag": 0.85,
    "ml_ranking": 0.90,
    "fine_tuning": 0.70,
    "ml_infrastructure": 0.65,
    "python_engineering": 0.75,
    "adjacent_nlp": 0.60,
}

# Titles that signal genuine AI/ML engineering work
POSITIVE_TITLE_SIGNALS = [
    "ml engineer", "machine learning engineer", "ai engineer", "nlp engineer",
    "research engineer", "applied scientist", "data scientist", "senior engineer",
    "software engineer", "search engineer", "ranking engineer", "rec sys",
    "recommendation", "backend engineer", "full stack engineer", "platform engineer",
    "ai specialist", "deep learning", "foundation model",
]

# Titles that are consulting-firm red flags when combined with only-consulting history
CONSULTING_DISQUALIFIER_PATTERNS = [
    "associate", "analyst", "consultant", "delivery manager",
    "program manager", "account manager",
]


def normalize_skill_name(skill: str) -> str:
    """Lowercase, strip, and normalize skill names."""
    return skill.lower().strip().replace("-", " ").replace("_", " ")


def skill_in_cluster(skill_name: str, cluster: list) -> bool:
    """Check if a normalized skill name matches any item in a cluster."""
    norm = normalize_skill_name(skill_name)
    return any(norm in c or c in norm for c in cluster)


def get_skill_cluster_coverage(candidate_skills: list) -> dict:
    """Get coverage score for each cluster based on candidate skills."""
    coverage = {}
    for cluster_name, cluster_terms in SKILL_CLUSTERS.items():
        matched = []
        for skill in candidate_skills:
            sname = normalize_skill_name(skill["name"])
            if any(sname in term or term in sname for term in cluster_terms):
                matched.append(skill)
        if matched:
            # Score based on best skill quality in cluster
            best_quality = max(skill_quality_score(s) for s in matched)
            coverage[cluster_name] = best_quality
        else:
            coverage[cluster_name] = 0.0
    return coverage


def skill_quality_score(skill: dict) -> float:
    """
    Score a single skill on quality — not just presence.
    Anti-keyword-stuffer: endorsements + duration + proficiency matter.
    """
    prof_map = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.65, "beginner": 0.35}
    prof_score = prof_map.get(skill.get("proficiency", "beginner"), 0.35)

    # Duration: logarithmic — 12 months = ~0.5, 36 months = ~0.75, 60+ months = 1.0
    duration = skill.get("duration_months", 0) or 0
    dur_score = min(1.0, math.log1p(duration) / math.log1p(60))

    # Endorsements: logarithmic cap at 50
    endorsements = skill.get("endorsements", 0) or 0
    end_score = min(1.0, math.log1p(endorsements) / math.log1p(50))

    # Weighted combination
    return 0.40 * prof_score + 0.35 * dur_score + 0.25 * end_score


def career_ai_depth(career_history: list) -> float:
    """
    Assess how much of the candidate's career involves actual AI/ML/IR work.
    This is the anti-title-stuffer check — skills must show up in career descriptions.
    """
    if not career_history:
        return 0.0

    ai_keywords = [
        "embedding", "vector", "retrieval", "ranking", "recommendation",
        "machine learning", "deep learning", "nlp", "language model", "llm",
        "transformer", "neural", "model", "training", "inference", "search",
        "similarity", "semantic", "rag", "fine-tun", "pytorch", "tensorflow",
        "hugging", "faiss", "elasticsearch", "opensearch", "bert", "gpt",
        "pipeline", "feature", "prediction", "classification", "regression",
    ]

    product_company_keywords = [
        "product", "startup", "saas", "platform", "users", "customers",
        "deployed", "production", "scale", "a/b test", "metric", "improvement",
        "shipped", "launched", "built and", "designed", "architected",
    ]

    total_months = 0
    weighted_ai_months = 0.0

    for role in career_history:
        desc = (role.get("description") or "").lower()
        duration = role.get("duration_months", 0) or 0
        total_months += duration

        # Count AI keyword hits
        ai_hits = sum(1 for kw in ai_keywords if kw in desc)
        product_hits = sum(1 for kw in product_company_keywords if kw in desc)

        # Title relevance
        title = (role.get("title") or "").lower()
        title_ai = any(kw in title for kw in ["ml", "ai", "nlp", "data", "engineer", "research", "scientist"])

        # AI-ness of this role
        if ai_hits >= 5 and title_ai:
            role_weight = 1.0
        elif ai_hits >= 3 or (ai_hits >= 2 and title_ai):
            role_weight = 0.7
        elif ai_hits >= 1:
            role_weight = 0.4
        else:
            role_weight = 0.1

        # Product company bonus
        if product_hits >= 2:
            role_weight = min(1.0, role_weight * 1.2)

        weighted_ai_months += duration * role_weight

    if total_months == 0:
        return 0.0

    return weighted_ai_months / total_months


def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Detect honeypot candidates with impossible profiles.
    Returns (is_honeypot, reason).
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check 1: Years of experience vs career history contradiction
    stated_yoe = profile.get("years_of_experience", 0)
    if career:
        actual_career_months = sum(r.get("duration_months", 0) or 0 for r in career)
        actual_yoe = actual_career_months / 12.0
        # If stated YOE is more than 2x actual career
        if stated_yoe > 3 and actual_yoe > 0 and stated_yoe > actual_yoe * 2.5:
            return True, f"YoE contradiction: stated {stated_yoe}yr but career history shows only {actual_yoe:.1f}yr"

    # Check 2: "Expert" in many skills with 0 endorsements and short duration
    expert_zero_end = [s for s in skills if s.get("proficiency") == "expert"
                       and s.get("endorsements", 0) == 0
                       and (s.get("duration_months") or 0) < 6]
    if len(expert_zero_end) >= 5:
        return True, f"Impossible skill profile: {len(expert_zero_end)} expert skills with 0 endorsements and <6mo duration"

    # Check 3: Future dates in career history
    today = date.today()
    for role in career:
        start = role.get("start_date")
        if start:
            try:
                start_date = datetime.strptime(start, "%Y-%m-%d").date()
                if start_date > today:
                    return True, f"Impossible career: start_date {start} is in the future"
            except Exception:
                pass

    # Check 4: Overlapping experiences with impossible company founding
    # (e.g., worked 8 years at company founded 3 years ago — hard to detect without
    # company founding data, but we check extreme durations at tiny companies)
    for role in career:
        duration = role.get("duration_months", 0) or 0
        company_size = role.get("company_size", "")
        # A 1-10 person company that someone worked at for >10 years is suspicious
        if duration > 120 and company_size == "1-10":
            return True, f"Suspicious: {duration}mo at a 1-10 person company"

    # Check 5: Skills list is impossibly broad for YOE
    if stated_yoe < 3 and len(skills) > 25:
        advanced_expert = [s for s in skills if s.get("proficiency") in ("advanced", "expert")]
        if len(advanced_expert) > 15:
            return True, f"Suspicious: {len(advanced_expert)} advanced/expert skills with only {stated_yoe}yr experience"

    return False, ""


def availability_score(signals: dict) -> float:
    """
    Score candidate's behavioral availability.
    A great-on-paper candidate who isn't reachable is useless.
    """
    score = 1.0

    # Recency of activity (key signal)
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_inactive = (date.today() - last_dt).days
            if days_inactive <= 7:
                activity_score = 1.0
            elif days_inactive <= 30:
                activity_score = 0.90
            elif days_inactive <= 60:
                activity_score = 0.75
            elif days_inactive <= 90:
                activity_score = 0.60
            elif days_inactive <= 180:
                activity_score = 0.40
            else:
                activity_score = 0.20
        except Exception:
            activity_score = 0.5
    else:
        activity_score = 0.5

    # Open to work flag
    open_flag = 1.1 if signals.get("open_to_work_flag") else 0.85

    # Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0.5) or 0.5
    # Non-linear: 0.8+ response rate = good, below 0.2 = very bad
    if response_rate >= 0.8:
        response_score = 1.0
    elif response_rate >= 0.5:
        response_score = 0.85
    elif response_rate >= 0.3:
        response_score = 0.65
    else:
        response_score = 0.40

    # Notice period (JD wants <30 days ideally)
    notice = signals.get("notice_period_days", 90)
    if notice <= 0:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.85
    elif notice <= 90:
        notice_score = 0.70
    elif notice <= 120:
        notice_score = 0.55
    else:
        notice_score = 0.40

    # Interview completion rate (reliable candidate)
    icr = signals.get("interview_completion_rate", 0.5) or 0.5
    icr_score = 0.5 + 0.5 * icr

    # Weighted combination
    availability = (
        0.35 * activity_score +
        0.25 * response_score +
        0.25 * notice_score +
        0.15 * icr_score
    ) * open_flag

    return min(1.0, availability)


def location_score(profile: dict, signals: dict) -> float:
    """Score location fit for the Pune/Noida-preferred role."""
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    preferred_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
                        "ncr", "gurugram", "gurgaon", "faridabad", "navi mumbai", "thane"]
    other_india = country == "india"

    if any(city in location for city in ["pune", "noida"]):
        return 1.0  # Perfect match
    elif any(city in location for city in preferred_cities):
        return 0.90
    elif other_india:
        return 0.75 if not willing_to_relocate else 0.82
    elif country in ("india",):
        return 0.75
    elif willing_to_relocate and country not in ("india",):
        return 0.55  # Willing but international
    else:
        return 0.30  # Outside India, not willing to relocate


def experience_fit_score(yoe: float) -> float:
    """Score years of experience fit for 5-9yr target range."""
    if 6 <= yoe <= 8:
        return 1.0
    elif 5 <= yoe <= 9:
        return 0.90
    elif 4 <= yoe < 5:
        return 0.75
    elif 9 < yoe <= 12:
        return 0.75
    elif 3 <= yoe < 4:
        return 0.55
    elif yoe > 12:
        return 0.60
    else:
        return 0.30


def is_consulting_only_career(career_history: list, signals: dict) -> bool:
    """Check if candidate's ENTIRE career is at big consulting firms (JD disqualifier)."""
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                        "mphasis", "hexaware", "l&t infotech", "hcl", "tech mahindra"]

    if not career_history:
        return False

    total_months = 0
    consulting_months = 0

    for role in career_history:
        company = (role.get("company") or "").lower()
        duration = role.get("duration_months", 0) or 0
        total_months += duration
        if any(firm in company for firm in consulting_firms):
            consulting_months += duration

    # If >85% of career at consulting firms
    if total_months > 0 and consulting_months / total_months > 0.85:
        return True
    return False


def title_career_corroboration(profile: dict, career_history: list, skills: list) -> float:
    """
    Key anti-stuffer check: do the listed skills appear in actual work descriptions?
    A Marketing Manager with 'RAG' and 'FAISS' skills = red flag.
    """
    current_title = (profile.get("current_title") or "").lower()

    # Hard disqualifier titles (non-technical roles claiming AI skills)
    hard_disq = ["marketing", "sales", "accountant", "hr manager", "human resources",
                 "customer support", "operations manager", "civil engineer",
                 "mechanical engineer", "electrical engineer", "content writer",
                 "graphic designer", "business development"]

    if any(disq in current_title for disq in hard_disq):
        return 0.15  # Very low — these people aren't engineers

    # Check how many claimed AI skills appear in career descriptions
    ai_skills = [s for s in skills
                 if any(normalize_skill_name(s["name"]) in t or t in normalize_skill_name(s["name"])
                        for cluster in SKILL_CLUSTERS.values() for t in cluster)]

    if not ai_skills:
        return 0.4  # No AI skills at all

    career_text = " ".join(
        (role.get("description") or "").lower() for role in career_history
    ).lower()

    corroborated = 0
    for skill in ai_skills:
        sname = normalize_skill_name(skill["name"])
        # Check if skill name or related term appears in career text
        if sname in career_text or any(term in career_text
                                        for cluster in SKILL_CLUSTERS.values()
                                        for term in cluster
                                        if sname in term or term in sname):
            corroborated += 1

    if len(ai_skills) == 0:
        return 0.5

    corroboration_ratio = corroborated / len(ai_skills)

    # Strong engineering title + corroborated skills = high score
    engineering_titles = ["engineer", "scientist", "researcher", "developer", "architect",
                          "specialist", "lead", "principal", "staff", "head of"]
    has_eng_title = any(t in current_title for t in engineering_titles)

    # AI/ML specific titles get automatic pass on title check
    ai_titles = ["ml engineer", "ai engineer", "nlp engineer", "machine learning",
                 "deep learning", "data scientist", "applied scientist", "research engineer",
                 "recommendation", "search engineer", "ranking engineer", "applied ml"]
    if any(t in current_title for t in ai_titles):
        has_eng_title = True
        # Boost corroboration for genuine AI titles
        corroboration_ratio = max(corroboration_ratio, 0.5)

    if has_eng_title and corroboration_ratio >= 0.6:
        return 1.0
    elif has_eng_title and corroboration_ratio >= 0.3:
        return 0.80
    elif corroboration_ratio >= 0.6:
        return 0.70
    elif corroboration_ratio >= 0.3:
        return 0.55
    else:
        return 0.35


def compute_skill_score(skills: list) -> float:
    """
    Compute overall skill fit score using cluster coverage with quality weighting.
    """
    coverage = get_skill_cluster_coverage(skills)

    weighted_sum = 0.0
    total_weight = 0.0

    for cluster_name, weight in CLUSTER_WEIGHTS.items():
        cluster_score = coverage.get(cluster_name, 0.0)
        weighted_sum += weight * cluster_score
        total_weight += weight

    if total_weight == 0:
        return 0.0

    raw_score = weighted_sum / total_weight

    # Bonus for covering both must-have clusters (core_retrieval + embedding_systems)
    if coverage.get("core_retrieval", 0) > 0.3 and coverage.get("embedding_systems", 0) > 0.3:
        raw_score = min(1.0, raw_score * 1.15)

    return min(1.0, raw_score)


def github_signal(signals: dict) -> float:
    """GitHub activity as a proxy for external validation (JD cares about this)."""
    score = signals.get("github_activity_score", -1)
    if score == -1:
        return 0.40  # No GitHub — slight negative
    elif score >= 80:
        return 1.0
    elif score >= 60:
        return 0.85
    elif score >= 40:
        return 0.70
    elif score >= 20:
        return 0.55
    elif score >= 1:
        return 0.45
    else:
        return 0.35


def education_score(education: list) -> float:
    """Score education — tier matters but isn't decisive."""
    if not education:
        return 0.50

    tier_scores = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.65, "tier_4": 0.50}
    best_score = max(tier_scores.get(edu.get("tier", "tier_4"), 0.5) for edu in education)

    # CS/Engineering field bonus
    cs_fields = ["computer science", "information technology", "software", "electronics",
                 "electrical", "statistics", "mathematics", "data science", "ai", "ml"]
    has_cs = any(
        any(f in (edu.get("field_of_study") or "").lower() for f in cs_fields)
        for edu in education
    )

    return min(1.0, best_score * (1.1 if has_cs else 1.0))


def platform_engagement_score(signals: dict) -> float:
    """Score platform engagement signals as a proxy for active job seeking."""
    # Profile completeness
    completeness = (signals.get("profile_completeness_score") or 50) / 100.0

    # Recruiter interest (saved by recruiters, search appearances)
    saved = min(1.0, math.log1p(signals.get("saved_by_recruiters_30d", 0) or 0) / math.log1p(20))
    search_app = min(1.0, math.log1p(signals.get("search_appearance_30d", 0) or 0) / math.log1p(500))
    apps = min(1.0, math.log1p(signals.get("applications_submitted_30d", 0) or 0) / math.log1p(10))

    # Verification signals
    verified = 0.0
    if signals.get("verified_email"):
        verified += 0.5
    if signals.get("verified_phone"):
        verified += 0.5

    return (0.30 * completeness + 0.20 * saved + 0.15 * search_app +
            0.15 * apps + 0.20 * verified)


def score_candidate(candidate: dict) -> dict:
    """
    Main scoring function. Returns a dict with all scores and the final score.
    """
    cid = candidate["candidate_id"]
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])

    # ── Step 1: Honeypot detection ──────────────────────────────────────────
    is_honeypot, honeypot_reason = detect_honeypot(candidate)
    if is_honeypot:
        return {
            "candidate_id": cid,
            "final_score": 0.0,
            "is_honeypot": True,
            "honeypot_reason": honeypot_reason,
            "pillar_scores": {},
        }

    # ── Step 2: Hard title disqualifier ─────────────────────────────────────
    title_corroboration = title_career_corroboration(profile, career, skills)

    # ── Step 3: Core pillar scores ───────────────────────────────────────────

    # Pillar A: Skill Fit (cluster-weighted quality)
    skill_fit = compute_skill_score(skills)

    # Pillar B: Career Depth in AI/ML (anti-keyword-stuffer)
    ai_career_depth = career_ai_depth(career)

    # Pillar C: Experience Fit (YoE range)
    yoe = profile.get("years_of_experience", 0) or 0
    exp_fit = experience_fit_score(yoe)

    # Pillar D: Location Fit
    loc_fit = location_score(profile, signals)

    # Pillar E: Behavioral Availability
    avail = availability_score(signals)

    # Supporting scores
    edu = education_score(education)
    github = github_signal(signals)
    platform = platform_engagement_score(signals)

    # Consulting-only penalty
    consulting_penalty = 0.70 if is_consulting_only_career(career, signals) else 1.0

    # ── Step 4: Weighted combination ────────────────────────────────────────
    # Weights reflect JD priorities:
    # Skill fit and career depth are most important
    # Availability matters a lot (the point of behavioral signals)
    # Location matters but isn't blocking
    technical_score = (
        0.35 * skill_fit +
        0.30 * ai_career_depth +
        0.15 * title_corroboration +
        0.10 * edu +
        0.10 * github
    )

    contextual_score = (
        0.50 * exp_fit +
        0.30 * loc_fit +
        0.20 * platform
    )

    # Behavioral availability is a multiplier on the combined score
    # (great skills + unavailable = lower priority)
    combined_score = (
        0.55 * technical_score +
        0.25 * contextual_score +
        0.20 * avail
    ) * consulting_penalty

    final_score = min(1.0, max(0.0, combined_score))

    return {
        "candidate_id": cid,
        "final_score": final_score,
        "is_honeypot": False,
        "pillar_scores": {
            "skill_fit": skill_fit,
            "ai_career_depth": ai_career_depth,
            "title_corroboration": title_corroboration,
            "exp_fit": exp_fit,
            "loc_fit": loc_fit,
            "availability": avail,
            "education": edu,
            "github": github,
            "platform": platform,
        },
        "meta": {
            "yoe": yoe,
            "title": profile.get("current_title", ""),
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
            "notice_days": signals.get("notice_period_days", 90),
            "last_active": signals.get("last_active_date", ""),
            "response_rate": signals.get("recruiter_response_rate", 0),
            "open_to_work": signals.get("open_to_work_flag", False),
            "willing_to_relocate": signals.get("willing_to_relocate", False),
            "is_consulting_only": is_consulting_only_career(career, signals),
        },
    }


def generate_reasoning(result: dict, candidate: dict) -> str:
    """
    Generate reasoning matching exact sample submission format:
    "{Title} with {X} yrs; {N} AI core skills; response rate {R}."
    """
    m = result["meta"]
    skills = candidate.get("skills", [])

    title = m["title"]
    yoe = m["yoe"]
    response_rate = m["response_rate"]

    # Count AI core skills matching the core clusters
    ai_core_terms = [
        term for cluster_name in ["core_retrieval", "embedding_systems", "llm_rag",
                                   "ml_ranking", "fine_tuning", "adjacent_nlp"]
        for term in SKILL_CLUSTERS[cluster_name]
    ]
    ai_skill_count = sum(
        1 for s in skills
        if any(t in normalize_skill_name(s["name"]) or normalize_skill_name(s["name"]) in t
               for t in ai_core_terms)
    )

    return (
        f"{title} with {yoe:.1f} yrs; "
        f"{ai_skill_count} AI core skills; "
        f"response rate {response_rate:.2f}."
    )


def rank_candidates(candidates_path: str, output_path: str, top_n: int = 100):
    """Main ranking pipeline."""
    print(f"Loading candidates from {candidates_path}...")

    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates. Scoring...")

    results = []
    honeypot_count = 0

    for i, candidate in enumerate(candidates):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{len(candidates)}...")
        result = score_candidate(candidate)
        result["_candidate"] = candidate  # Keep reference for reasoning
        results.append(result)
        if result["is_honeypot"]:
            honeypot_count += 1

    print(f"Scoring complete. Honeypots detected: {honeypot_count}")

    # Sort by final score descending (rounded to 4dp to match CSV), then candidate_id ascending
    valid_results = [r for r in results if not r["is_honeypot"]]
    valid_results.sort(key=lambda r: (-round(r["final_score"], 4), r["candidate_id"]))

    # Take top 100
    top_100 = valid_results[:top_n]

    print(f"\nTop {top_n} candidates selected. Generating CSV...")

    # Write output CSV
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, result in enumerate(top_100, start=1):
            cid = result["candidate_id"]
            score = round(result["final_score"], 4)
            candidate = result["_candidate"]
            reasoning = generate_reasoning(result, candidate)
            writer.writerow([cid, rank_idx, score, reasoning])

    print(f"Output written to {output_path}")

    # Print top 10 for inspection
    print("\n=== TOP 10 CANDIDATES ===")
    for i, r in enumerate(top_100[:10]):
        m = r["meta"]
        p = r["pillar_scores"]
        print(f"#{i+1} {r['candidate_id']} | Score: {r['final_score']:.4f}")
        print(f"     {m['title']} | {m['yoe']}yr | {m['location']}, {m['country']}")
        print(f"     SkillFit={p['skill_fit']:.2f} CareerDepth={p['ai_career_depth']:.2f} "
              f"Avail={p['availability']:.2f} LocFit={p['loc_fit']:.2f}")
        print()

    return top_100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl file")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of top candidates to output")
    args = parser.parse_args()

    import time
    start = time.time()
    rank_candidates(args.candidates, args.out, args.top)
    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s")
