# LabAgentSkill — Project Overview

LabAgentSkill is a **skill-driven agent experimentation workspace** focused on evaluating agent behaviors with experiments in the industrial usecase to verify if SLMs can effectively leverage skills to solve complex tasks.

<div style="text-align: center;">
<img src="logo_agent_skill.png" style="width: 50%; height: auto;" alt="LabAgentSkill Logo">
</div>

## Key Features

- **Three-stage skill pipeline** — Select → Discover → Execute: the agent first identifies relevant skills, discovers any referenced sub-skills, then executes the task following skill instructions.
- **Multi-backend LLM support** — OpenAI (GPT-4o-mini), Anthropic (Claude), Google (Gemma), Qwen, and local vLLM servers.
- **Three industry benchmarks** — IMDB sentiment analysis, FiNER-139 XBRL tag classification, and InsurBench insurance email triage.
- **Three evaluation approaches per sample** — Skill-based pipeline, simple baseline (no skills), and full skill-context injection (RAG-style).
- **Skill Hub** — A filesystem-based repository of reusable SKILL.md files that define step-by-step reasoning workflows.
- **Automatic context management** — Message trimming middleware and retry-with-truncation for staying within LLM context windows.

---

# Project Structure

```
LabAgentSkill/
├── main.py                        # Entry point (placeholder)
├── pyproject.toml                 # Project metadata and dependencies
├── README.md
│
├── src/LabAgentSkill/             # Core Python package
│   ├── __init__.py
│   ├── SkillAwareAgent.py         # LangGraph-based agent with skill awareness,
│   │                              #   chat history, and message trimming
│   ├── evaluate.py                # Label extraction / prediction parsing for
│   │                              #   all three benchmarks (IMDB, FiNER, InsurBench)
│   └── skills_utils/              # Skill management utilities
│       ├── __init__.py            # Public API re-exports
│       ├── parser.py              # SKILL.md frontmatter parser (YAML + Markdown)
│       ├── models.py              # SkillProperties data class
│       ├── validator.py           # Skill validation logic
│       ├── prompt.py              # Skill-to-prompt conversion
│       ├── skill_util.py          # Bulk read, display, parse, sample skills
│       ├── cli.py                 # CLI helpers
│       └── errors.py              # Custom exceptions (SkillError, ParseError, etc.)
│
├── prompts/                       # Jinja2 prompt templates
│   ├── p_skill_select.jinja       # Step 1: Skill selection from available list
│   ├── p_skill_discov.jinja       # Step 2: Discover sub-skills referenced in selected skills
│   ├── p_skill_exec.jinja         # Step 3: Execute task using loaded skill instructions
│   ├── p_default_system.jinja     # Default system prompt (baseline, no skills)
│   ├── p_exec_imdb.jinja          # Task prompt: IMDB sentiment classification
│   ├── p_exec_finer.jinja         # Task prompt: FiNER-139 XBRL tag classification
│   ├── p_exec_insurBench.jinja    # Task prompt: InsurBench email action decision
│   └── system_prompt.jinja        # General system prompt template
│
├── skillsHub/                     # Skill repository (SKILL.md files)
│   ├── skills/                    # IMDB benchmark skills (5 skills)
│   ├── skills_finer/              # FiNER benchmark skills (6 skills)
│   ├── skills_insurBench/         # InsurBench benchmark skills (7 skills)
│   ├── skills_scaling/            # Large-scale skill pool (130+ skills)
│   └── skillTemp/                 # Temporary / work-in-progress skills
│
├── scripts/                       # Standalone benchmark execution scripts
│   ├── test_agent_finer_300_stadard-claude.py   # FiNER benchmark with Claude
│   └── visualize_agent_result.py                # HTML visualization of agent results
│
├── notebooks/                     # Jupyter experiment notebooks (22 notebooks)
│   ├── test_agent_imdb_300_*.ipynb              # IMDB experiments across models
│   ├── test_agent_finer_300_*.ipynb             # FiNER experiments across models
│   ├── test_agent_insurbench_300_*.ipynb        # InsurBench experiments
│   ├── test_agent_skill_scaling_standard.ipynb  # Skill scaling experiment
│   ├── test_agent_multi_run.ipynb               # Multi-run aggregation
│   └── test_dataset_*.ipynb                     # Dataset exploration notebooks
│
├── assets/
│   ├── datasets/                  # Raw benchmark datasets
│   │   ├── aclImdb/               # IMDB movie review dataset
│   │   ├── finer-139/             # FiNER-139 financial NER dataset
│   │   └── insureBench.jsonl      # InsurBench insurance email dataset
│   ├── results/                   # JSONL output files from experiments
│   └── photos/                    # Project images and logos
│
└── refs/                          # Reference implementations and documentation
```

---

# Agent Pipeline

The core evaluation pipeline follows a **three-step skill-augmented approach**:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Step 1: Select  │────▶│  Step 2: Discover │────▶│  Step 3: Execute │
│                  │     │                   │     │                  │
│ Given the task + │     │ Read selected     │     │ Inject all skill │
│ list of skills,  │     │ skill bodies to   │     │ instructions as  │
│ pick relevant    │     │ find referenced   │     │ system context,  │
│ ones by name +   │     │ sub-skills.       │     │ then solve the   │
│ description.     │     │ Repeat until no   │     │ task following    │
│                  │     │ new skills found. │     │ the workflow.     │
│ Output: JSON     │     │ Output: JSON      │     │ Output: JSON     │
│ {"Skills": [...]}│     │ {"Skills": [...]} │     │ {"Message": "..."}│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

Each sample is also evaluated with two additional baselines for comparison:
- **Simple baseline** — The agent receives only the default system prompt and the task, with no skill context.
- **Full skill context** — All skill instructions are injected directly into the user prompt (RAG-style), bypassing the selection pipeline.

---

# SKILL.md Format

Each skill is defined as a `SKILL.md` file inside its own subdirectory under `skillsHub/`. The file uses YAML frontmatter for metadata and Markdown for the instruction body:

```markdown
---
name: my-skill-name
description: A one-line description of what this skill does.
---

## Instructions

Step-by-step reasoning workflow, classification rules, examples, etc.
```

**Required fields:** `name`, `description`
**Optional fields:** `license`, `compatibility`, `allowed_tools`, `metadata`

---

# Additional Information

## InsurBench Example

This dataset is completely close domain and requires understanding of insurance email threads, policy details, and action urgency. The agent must analyze the latest email in the thread while considering relevant historical context to determine if immediate action is required. And we just show several filtered examples in order to make sure there is no data and private information leakage.  

**Task:** Given a multi-turn insurance email thread, determine whether the insurance company must take action NOW based on the latest email.

**Example Dataset:** `assets/datasets/insureBench.jsonl` — each record contains an email history and a ground-truth YES/NO label.

**Skill used:** `insurance-mail-triage` — a 4-step process: extract the latest message → keep essential history → classify intent → decide action (YES / NO).

**Evaluation:** The agent's response is parsed for "yes" or "no" using word-boundary regex (`get_insurBench_predicted_label`) to avoid false positives from substrings like "yesterday".

## IMDB Example

**Task:** Classify a movie review as **positive** or **negative** (binary sentiment analysis).

**Dataset:** ACL IMDB movie review dataset (`assets/datasets/aclImdb/`), sampled to 300 examples.

**Skill used:** `movie-sentiment-analysis` — a reasoning chain: identify sentiment cues → analyze modifiers → weight aspects → conclude with a label.

**Evaluation:** The agent's response is parsed via heuristic rules (`get_predicted_label`): if only "positive" appears → positive; if only "negative" → negative; if ambiguous, use whichever appears first.

## FiNER Example

**Task:** Given a sentence from a financial SEC filing and a target numeric entity, classify the entity into one of **139 XBRL taxonomy tags** (e.g., `DebtInstrumentFaceAmount`, `Revenues`, `Goodwill`).

**Dataset:** `Volavion/finer-139-numeric-sampled` from Hugging Face Hub — sentences with BIO-tagged financial entities.

**Skill used:** `XBRL-tag-classification` — detailed tag definitions organized by category (interest rates, debt amounts, equity, etc.) with disambiguation rules and examples.

**Evaluation:** The agent's response is matched against all 139 known tags using substring matching (`get_prediction_XBRL_TAGS`). Both exact and case-insensitive matching are tried, always preferring the longest match to avoid partial hits.

---
