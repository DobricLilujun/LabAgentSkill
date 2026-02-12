# Setup: Load environment variables and dependencies
import os
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

project_root = Path.cwd()
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"✓ Added to sys.path: {src_path}")

from LabAgentSkill import skills_utils
from LabAgentSkill.SkillAwareAgent import SkillAwareAgent

root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
env_path = root_dir / ".env"
env = {}

if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

# Set API key
os.environ["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
env = Environment(loader=FileSystemLoader('prompts/'))  
skills_folder = Path("/home/snt/projects_lujun/LabAgentSkill/skillsHub/skills_finer")
all_skills = skills_utils.read_all_skills_metadata(skills_folder)
for skill in all_skills:
    print(f"  - {skill['name']}: {skill['description']}")

# model_name = "gpt-4o-mini"
# base_url = None
# model_name = "google/gemma-3-270m-it"
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# base_url = "http://127.0.0.1:8001/v1"

base_url = "http://10.6.32.18:8000/v1"

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

from LabAgentSkill.evaluate import get_predicted_label, get_prediction_XBRL_TAGS

print(f"Using model: {model_name}")
# Initialize agents
agent_skill_aware = SkillAwareAgent(use_chat_history=True, use_trim_messages=True, model=model_name, base_url=base_url, max_tokens=512)
agent_skill_exec_agent = SkillAwareAgent(use_chat_history=True, use_trim_messages=True, model=model_name, base_url=base_url, max_tokens=512)
agent_simple = SkillAwareAgent(use_chat_history=True, use_trim_messages=False, model=model_name, base_url=base_url, max_tokens=512)

p_exec_finer_temp = env.get_template('p_exec_finer.jinja')
p_skill_select_temp = env.get_template('p_skill_select.jinja')
p_skill_discov_temp = env.get_template('p_skill_discov.jinja')
p_default_system_temp = env.get_template('p_default_system.jinja')
p_skill_exec_temp = env.get_template('p_skill_exec.jinja')

# JSONL output path
output_dir = "/home/snt/projects_lujun/LabAgentSkill/assets/results/"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
jsonl_path = output_dir+f"finer_standard_{model_name.split('/')[-1]}_{timestamp}.jsonl"
print(f"Results will be saved to: {jsonl_path}")

skill_count = 0
count_row = 0
if os.path.exists(jsonl_path):
    df_exist = pd.read_json(jsonl_path, lines=True)
    count_row = len(df_exist)
    print(f"Resume from row: {count_row}")


# Process each sample
for idx, row in tqdm( loaded_df.iterrows(), total=len(loaded_df), desc="Processing samples",):

    if idx < count_row:
        continue
    sample_start_time, sentence, tag_name, tag_token = time.time(), row.sentence, row.tag_name[2:], row.tag_token
    true_label = tag_name
    # Step 1: Skill Selection
    # print(f"Start Skill Selection Phase for Sample {idx + 1}/{len(loaded_df)}")
    skill_context = "\n".join([
        f"- **{skill['name']}**: {skill['description']}"
        for skill in all_skills
    ])

    p_skill_select = p_skill_select_temp.render(SKILL_CONTEXT=skill_context)
    p_exec_finer = p_exec_finer_temp.render(SENTENCE_CONTENT = sentence, NUMERIC_ENTITY = tag_token)
    skill_select_resp = agent_skill_aware.chat(user_input=p_exec_finer, custom_system_prompt=p_skill_select)
    selected_skills = skills_utils.parse_skills_from_json_response(json_response=skill_select_resp, skills_hub_dir=skills_folder)

    # Track whether "movie-sentiment-analysis" was selected in Step 1 
    selected_skill_names_step1 = [s["name"] for s in selected_skills]
    hit_target_skill = "XBRL-tag-classification" in selected_skill_names_step1 ## This is hard Coded

    skill_execution_context = ""
    for skill_meta in selected_skills:
        skill_execution_context += (
            f"SKill {skill_count + 1}: \n"
            f"{skill_meta['description']}\n"
            f"{'\n'.join(skill_meta['body'].split('\n')[1:])}\n\n"
        )
        skill_count += 1

    skill_count_prev = skill_count

    # Step 2: Skill Discovery
    discovery_rounds = 0
    while len(selected_skills) > 0:
        p_skill_discov = p_skill_discov_temp.render(SKILL_CONTEXT=skill_execution_context)
        skill_discov_resp = agent_skill_exec_agent.chat(user_input=p_skill_discov, custom_system_prompt=p_default_system_temp.render())
        selected_skills = skills_utils.parse_skills_from_json_response(json_response=skill_discov_resp, skills_hub_dir=skills_folder)

        for skill_meta in selected_skills:
            skill_execution_context += (
                f"SKill {skill_count + 1}: \n"
                f"{skill_meta['description']}\n"
                f"{'\n'.join(skill_meta['body'].split('\n')[1:])}\n\n"
            )
            skill_count += 1
        discovery_rounds += 1
    new_skills_found = skill_count - skill_count_prev


    # print(f"End of skill discovery phase. Found total of new skills: {new_skills_found}")
    # Step 3: Query Execution
    p_exec_finer_sys = p_skill_exec_temp.render(SKILL_CONTEXT=skill_execution_context)
    finer_exec_response = agent_skill_exec_agent.chat(user_input=p_exec_finer, custom_system_prompt=p_exec_finer_sys)
    message_classification = skills_utils.parse_message_from_json_response(finer_exec_response)
    is_correct = tag_name.lower() in message_classification.strip().lower()

    predicted_label = get_prediction_XBRL_TAGS(message_classification)
    sample_end_time = time.time()
    sample_elapsed = sample_end_time - sample_start_time
    chat_history_agent_skill_select = agent_skill_aware.get_human_ai_message_history()
    chat_history_agent_exec = agent_skill_exec_agent.get_human_ai_message_history()

    # Build record and append to JSONL
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
        "task_type": "agent_skill_based"
        
    }

    dataframe_record = pd.DataFrame([record])
    dataframe_record.to_json(jsonl_path, orient="records", lines=True, mode="a" if os.path.exists(jsonl_path) else "w")
    agent_skill_aware.clear_history()
    agent_skill_exec_agent.clear_history()

    ######################################################################################################################################
    sample_start_time, sentence, tag_name, tag_token = time.time(), row.sentence, row.tag_name[2:], row.tag_token
    p_exec_finer_sys = p_default_system_temp.render()
    finer_exec_response = agent_simple.chat(user_input=p_exec_finer, custom_system_prompt=p_exec_finer_sys)
    message_classification = skills_utils.parse_message_from_json_response(finer_exec_response)
    is_correct = tag_name.lower() in message_classification.strip().lower()

    predicted_label = get_prediction_XBRL_TAGS(message_classification)
    sample_end_time = time.time()
    sample_elapsed = sample_end_time - sample_start_time
    chat_history_agent_exec = agent_simple.get_human_ai_message_history()

    # Build record and append to JSONL
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
        "task_type": "agent_simple"
        
    }

    dataframe_record = pd.DataFrame([record])
    dataframe_record.to_json(jsonl_path, orient="records", lines=True, mode="a" if os.path.exists(jsonl_path) else "w")
    agent_simple.clear_history()


    ######################################################################################################################################
    sample_start_time, sentence, tag_name, tag_token = time.time(), row.sentence, row.tag_name[2:], row.tag_token
    skill_context_all =  "The following are skills informaiton you can use as a reference for user request:\n".join([
        f"- **{skill['name']}**:\n {skill['description']} **:\n {skill['body']}"
        for skill in all_skills
    ])
    p_exec_finer = p_exec_finer_temp.render(SENTENCE_CONTENT = sentence, NUMERIC_ENTITY = tag_token+  "\n" + skill_context_all)
    p_exec_finer_sys = p_default_system_temp.render()

    finer_exec_response = agent_simple.chat(user_input=p_exec_finer, custom_system_prompt=p_exec_finer_sys)
    message_classification = skills_utils.parse_message_from_json_response(finer_exec_response)
    is_correct = true_label.lower() in message_classification.strip().lower()

    predicted_label = get_prediction_XBRL_TAGS(message_classification)
    sample_end_time = time.time()
    sample_elapsed = sample_end_time - sample_start_time
    chat_history_agent_exec = agent_simple.get_human_ai_message_history()

    # Build record and append to JSONL
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
        "task_type": "agent_skill_full_context"
        
    }

    dataframe_record = pd.DataFrame([record])
    dataframe_record.to_json(jsonl_path, orient="records", lines=True, mode="a" if os.path.exists(jsonl_path) else "w")
    agent_simple.clear_history()

print(f"\n{'='*60}")
print(f"All {len(loaded_df)} samples processed. Results saved to: {jsonl_path}")
print(f"{'='*60}")