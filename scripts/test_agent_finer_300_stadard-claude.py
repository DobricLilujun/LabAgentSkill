# =============================================================================
# FiNER-139 XBRL Tag Classification Benchmark Script (Claude Model)
# =============================================================================
# This script evaluates an LLM-based agent system on the FiNER-139 dataset
# for financial named entity recognition (NER) / XBRL tag classification.
# It runs THREE approaches per sample and saves all results to a JSONL file:
#   1. Agent with skill selection + discovery pipeline (agent_skill_based)
#   2. Simple agent without any skills (agent_simple)
#   3. Agent with full skill context injected directly (agent_skill_full_context)
# =============================================================================

# --- Standard library and third-party imports ---
import os
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# --- Add the project's src/ directory to sys.path so local packages can be imported ---
project_root = Path.cwd()
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"✓ Added to sys.path: {src_path}")

# --- Import custom modules for skill management and the skill-aware agent ---
from LabAgentSkill import skills_utils
from LabAgentSkill.SkillAwareAgent import SkillAwareAgent

# --- Load environment variables from .env file ---
# Handle both running from project root or from the notebooks/ subdirectory
root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
env_path = root_dir / ".env"
env = {}

# Manually parse the .env file (skipping comments and blank lines)
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

# Set the Anthropic API key from .env or fall back to existing environment variable
os.environ["ANTHROPIC_API_KEY"] = env.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

# --- Initialize Jinja2 template environment for loading prompt templates ---
env = Environment(loader=FileSystemLoader('prompts/'))  

# --- Load all available skills from the FiNER skills hub ---
skills_folder = Path("/home/snt/projects_lujun/LabAgentSkill/skillsHub/skills_finer")
all_skills = skills_utils.read_all_skills_metadata(skills_folder)
for skill in all_skills:
    print(f"  - {skill['name']}: {skill['description']}")

# --- Model configuration ---
# Using Claude Opus 4.6 via Anthropic API (no custom base_url needed)
model_name = "claude-opus-4-6"
base_url = None
# Alternative local model configurations (commented out):
# model_name = "google/gemma-3-270m-it"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# base_url = "http://127.0.0.1:8001/v1"


# --- Load the FiNER-139 dataset from Hugging Face Hub ---
# This is a sampled subset of the FiNER-139 dataset containing numeric financial entities
from datasets import load_dataset
dataset_name = "Volavion/finer-139-numeric-sampled"
loaded_dataset = load_dataset(dataset_name, split="train")
loaded_df = loaded_dataset.to_pandas()
loaded_df = loaded_df.reset_index(drop=True)


import time
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Import evaluation utilities for parsing XBRL tag predictions
from LabAgentSkill.evaluate import get_predicted_label, get_prediction_XBRL_TAGS

print(f"Using model: {model_name}")

# --- Initialize three separate agent instances for different evaluation modes ---
# Agent 1: Skill-aware agent for skill selection (Step 1) — uses chat history with trimming
agent_skill_aware = SkillAwareAgent(use_chat_history=True, use_trim_messages=True, model=model_name, base_url=base_url, max_tokens=256)
# Agent 2: Skill execution agent for skill discovery (Step 2) and final query (Step 3)
agent_skill_exec_agent = SkillAwareAgent(use_chat_history=True, use_trim_messages=True, model=model_name, base_url=base_url, max_tokens=256)
# Agent 3: Simple baseline agent — no skill pipeline, no message trimming
agent_simple = SkillAwareAgent(use_chat_history=True, use_trim_messages=False, model=model_name, base_url=base_url, max_tokens=256)

# --- Load Jinja2 prompt templates ---
p_exec_finer_temp = env.get_template('p_exec_finer.jinja')       # Task prompt: classify a numeric entity in a sentence
p_skill_select_temp = env.get_template('p_skill_select.jinja')   # System prompt: instruct agent to select relevant skills
p_skill_discov_temp = env.get_template('p_skill_discov.jinja')   # Prompt: discover additional skills from execution context
p_default_system_temp = env.get_template('p_default_system.jinja')  # Default system prompt (no skills)
p_skill_exec_temp = env.get_template('p_skill_exec.jinja')       # System prompt: execute task using selected skills

# --- Configure JSONL output path with model name and timestamp ---
output_dir = "/home/snt/projects_lujun/LabAgentSkill/assets/results/"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
jsonl_path = output_dir+f"finer_standard_{model_name.split('/')[-1]}_{timestamp}.jsonl"
print(f"Results will be saved to: {jsonl_path}")

# --- Resume support: skip already-processed rows if the output file exists ---
skill_count = 0
count_row = 0
if os.path.exists(jsonl_path):
    df_exist = pd.read_json(jsonl_path, lines=True)
    count_row = len(df_exist)
    print(f"Resume from row: {count_row}")


# =============================================================================
# Main Processing Loop — iterate over every sample in the dataset
# For each sample, three evaluation approaches are executed sequentially.
# =============================================================================
for idx, row in tqdm( loaded_df.iterrows(), total=len(loaded_df), desc="Processing samples",):

    # Skip already-processed rows (for resume support)
    if idx < count_row:
        continue

    # Extract fields from the current row; strip the leading "B-" or "I-" BIO prefix from tag_name
    sample_start_time, sentence, tag_name, tag_token = time.time(), row.sentence, row.tag_name[2:], row.tag_token
    true_label = tag_name

    # =====================================================================
    # APPROACH 1: Agent with Skill Selection + Discovery Pipeline
    # =====================================================================

    # --- Step 1: Skill Selection ---
    # Build a formatted list of all available skills for the agent to choose from
    skill_context = "\n".join([
        f"- **{skill['name']}**: {skill['description']}"
        for skill in all_skills
    ])

    # Render the skill selection system prompt and the FiNER task user prompt
    p_skill_select = p_skill_select_temp.render(SKILL_CONTEXT=skill_context)
    p_exec_finer = p_exec_finer_temp.render(SENTENCE_CONTENT = sentence, NUMERIC_ENTITY = tag_token)

    # Ask the skill-aware agent to select relevant skills for this task
    skill_select_resp = agent_skill_aware.chat(user_input=p_exec_finer, custom_system_prompt=p_skill_select)
    selected_skills = skills_utils.parse_skills_from_json_response(json_response=skill_select_resp, skills_hub_dir=skills_folder)

    # Check if the target skill "XBRL-tag-classification" was among the selected skills
    selected_skill_names_step1 = [s["name"] for s in selected_skills]
    hit_target_skill = "XBRL-tag-classification".lower() in selected_skill_names_step1  # Hard-coded target skill name

    # Build the skill execution context by concatenating descriptions and bodies
    # of all selected skills (skipping the first line of each body, which is typically a header)
    skill_execution_context = ""
    for skill_meta in selected_skills:
        skill_execution_context += (
            f"SKill {skill_count + 1}: \n"
            f"{skill_meta['description']}\n"
            f"{'\n'.join(skill_meta['body'].split('\n')[1:])}\n\n"
        )
        skill_count += 1

    skill_count_prev = skill_count

    # --- Step 2: Skill Discovery ---
    # Iteratively discover new skills referenced within the selected skills.
    # The loop continues until no further new skills are found.
    discovery_rounds = 0
    while len(selected_skills) > 0:
        p_skill_discov = p_skill_discov_temp.render(SKILL_CONTEXT=skill_execution_context)
        skill_discov_resp = agent_skill_exec_agent.chat(user_input=p_skill_discov, custom_system_prompt=p_default_system_temp.render())
        selected_skills = skills_utils.parse_skills_from_json_response(json_response=skill_discov_resp, skills_hub_dir=skills_folder)

        # Append any newly discovered skills to the execution context
        for skill_meta in selected_skills:
            skill_execution_context += (
                f"SKill {skill_count + 1}: \n"
                f"{skill_meta['description']}\n"
                f"{'\n'.join(skill_meta['body'].split('\n')[1:])}\n\n"
            )
            skill_count += 1
        discovery_rounds += 1
    new_skills_found = skill_count - skill_count_prev  # Total skills discovered in Step 2


    # --- Step 3: Query Execution with skill context ---
    # Use the execution agent with all gathered skill context to classify the entity
    p_exec_finer_sys = p_skill_exec_temp.render(SKILL_CONTEXT=skill_execution_context)
    finer_exec_response = agent_skill_exec_agent.chat(user_input=p_exec_finer, custom_system_prompt=p_exec_finer_sys)
    message_classification = skills_utils.parse_message_from_json_response(finer_exec_response)
    is_correct = tag_name.lower() in message_classification.strip().lower()

    # Extract the structured XBRL tag prediction from the raw response
    predicted_label = get_prediction_XBRL_TAGS(message_classification)
    sample_end_time = time.time()
    sample_elapsed = sample_end_time - sample_start_time

    # Capture chat histories for reproducibility and analysis
    chat_history_agent_skill_select = agent_skill_aware.get_human_ai_message_history()
    chat_history_agent_exec = agent_skill_exec_agent.get_human_ai_message_history()

    # Build a result record and append it to the JSONL output file
    record = {
        "index": int(idx),
        "sentence": sentence,
        "tag_name": tag_name,
        "tag_token": tag_token,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "raw_response": message_classification,
        "correct": is_correct,
        "selected_skills_step1": selected_skill_names_step1,
        "hit_target_skill": hit_target_skill,
        "new_skills_discovered": new_skills_found,
        "discovery_rounds": discovery_rounds,
        "elapsed_seconds": round(sample_elapsed, 4),
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "chat_history_agent_skill_select": chat_history_agent_skill_select,
        "chat_history_agent_exec": chat_history_agent_exec,
        "task_type": "agent_skill_based"  # Label: skill-based pipeline
        
    }

    dataframe_record = pd.DataFrame([record])
    dataframe_record.to_json(jsonl_path, orient="records", lines=True, mode="a" if os.path.exists(jsonl_path) else "w")

    # Clear agent histories before the next approach to prevent cross-contamination
    agent_skill_aware.clear_history()
    agent_skill_exec_agent.clear_history()

    # =====================================================================
    # APPROACH 2: Simple Agent Baseline (no skills)
    # =====================================================================
    # This serves as a baseline — the agent receives only the default system
    # prompt and the task prompt, with no skill context whatsoever.
    sample_start_time, sentence, tag_name, tag_token = time.time(), row.sentence, row.tag_name[2:], row.tag_token
    p_exec_finer_sys = p_default_system_temp.render()
    finer_exec_response = agent_simple.chat(user_input=p_exec_finer, custom_system_prompt=p_exec_finer_sys)
    message_classification = skills_utils.parse_message_from_json_response(finer_exec_response)
    is_correct = tag_name.lower() in message_classification.strip().lower()

    predicted_label = get_prediction_XBRL_TAGS(message_classification)
    sample_end_time = time.time()
    sample_elapsed = sample_end_time - sample_start_time
    chat_history_agent_exec = agent_simple.get_human_ai_message_history()

    # Build record — skill-related fields are empty since no skills were used
    record = {
        "index": int(idx),
        "sentence": sentence,
        "tag_name": tag_name,
        "tag_token": tag_token,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "raw_response": message_classification,
        "correct": is_correct,
        "selected_skills_step1": "",
        "hit_target_skill": "",
        "new_skills_discovered": "",
        "discovery_rounds": "",
        "elapsed_seconds": round(sample_elapsed, 4),
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "chat_history_agent_skill_select": "",
        "chat_history_agent_exec": chat_history_agent_exec,
        "task_type": "agent_simple"  # Label: simple baseline (no skills)
        
    }

    dataframe_record = pd.DataFrame([record])
    dataframe_record.to_json(jsonl_path, orient="records", lines=True, mode="a" if os.path.exists(jsonl_path) else "w")
    agent_simple.clear_history()


    # =====================================================================
    # APPROACH 3: Agent with Full Skill Context (RAG-style injection)
    # =====================================================================
    # Instead of using the skill selection/discovery pipeline, ALL skills
    # (names, descriptions, and full bodies) are directly injected into the
    # user prompt as additional context. This tests whether giving the model
    # all skill information upfront performs better than the dynamic pipeline.
    sample_start_time, sentence, tag_name, tag_token = time.time(), row.sentence, row.tag_name[2:], row.tag_token

    # Concatenate all skill metadata (name + description + body) into one context string
    skill_context_all =  "The following are skills informaiton you can use as a reference for user request:\n".join([
        f"- **{skill['name']}**:\n {skill['description']} **:\n {skill['body']}"
        for skill in all_skills
    ])

    # Append full skill context directly to the task prompt
    p_exec_finer = p_exec_finer_temp.render(SENTENCE_CONTENT = sentence, NUMERIC_ENTITY = tag_token+  "\n" + skill_context_all)
    p_exec_finer_sys = p_default_system_temp.render()

    finer_exec_response = agent_simple.chat(user_input=p_exec_finer, custom_system_prompt=p_exec_finer_sys)
    message_classification = skills_utils.parse_message_from_json_response(finer_exec_response)
    is_correct = true_label.lower() in message_classification.strip().lower()

    predicted_label = get_prediction_XBRL_TAGS(message_classification)
    sample_end_time = time.time()
    sample_elapsed = sample_end_time - sample_start_time
    chat_history_agent_exec = agent_simple.get_human_ai_message_history()

    # Build record — skill-related fields are empty (skills were injected, not selected)
    record = {
        "index": int(idx),
        "sentence": sentence,
        "tag_name": tag_name,
        "tag_token": tag_token,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "raw_response": message_classification,
        "correct": is_correct,
        "selected_skills_step1": "",
        "hit_target_skill": "",
        "new_skills_discovered": "",
        "discovery_rounds": "",
        "elapsed_seconds": round(sample_elapsed, 4),
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "chat_history_agent_skill_select": "",
        "chat_history_agent_exec": chat_history_agent_exec,
        "task_type": "agent_skill_full_context"  # Label: full skill context injection
        
    }

    dataframe_record = pd.DataFrame([record])
    dataframe_record.to_json(jsonl_path, orient="records", lines=True, mode="a" if os.path.exists(jsonl_path) else "w")
    agent_simple.clear_history()

# --- Final summary ---
print(f"\n{'='*60}")
print(f"All {len(loaded_df)} samples processed. Results saved to: {jsonl_path}")
print(f"{'='*60}")