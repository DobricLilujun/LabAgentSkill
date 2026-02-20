# =============================================================================
# Skill Utility Functions
# =============================================================================
# This module provides helper functions for managing and interacting with
# "skills" stored as SKILL.md files on the filesystem. Key capabilities:
#   - List and display available skills in a directory
#   - Read skill metadata (name, description, body, etc.) into dicts
#   - Parse skill names from LLM JSON responses and match to loaded skills
#   - Extract message content from JSON agent responses
#   - Random skill sampling for experiments
#   - Text replacement utility for swapping "skill" terminology
# =============================================================================

from pathlib import Path
from .parser import read_properties, find_skill_md


def list_skills_in_folder(skills_folder: str) -> None:
    """
    Print a formatted listing of all skills found in a directory.

    Scans ``skills_folder`` for subdirectories containing a SKILL.md file,
    reads each skill's name and description, and prints them as a numbered list.
    Long descriptions are truncated to 70 characters.

    Args:
        skills_folder: Path to the folder containing skill subdirectories.
    """
    skills_path = Path(skills_folder).resolve()
    
    if not skills_path.exists():
        print(f"❌ Folder not found: {skills_path}")
        return
    
    # Collect directories that contain a SKILL.md file (i.e., valid skill dirs)
    skill_dirs = []
    for item in sorted(skills_path.iterdir()):
        if item.is_dir() and find_skill_md(item):
            skill_dirs.append(item)
    
    if not skill_dirs:
        print(f"ℹ️  No skills found in: {skills_path}")
        return
    
    # Print a formatted header and numbered skill list
    print("\n" + "=" * 80)
    print(f"SKILLS IN: {skills_path.name}")
    print("=" * 80)
    print(f"Total: {len(skill_dirs)} skills\n")
    
    for idx, skill_dir in enumerate(skill_dirs, 1):
        try:
            props = read_properties(skill_dir)
            name = props.name
            description = props.description
            
            # Truncate long descriptions to keep output tidy
            if len(description) > 70:
                description = description[:67] + "..."
            
            print(f"[{idx}] {name}")
            print(f"    └─ {description}")
            
        except Exception as e:
            print(f"[{idx}] ❌ Error reading {skill_dir.name}: {e}")
    
    print("\n" + "=" * 80 + "\n")


# --- Re-import (safe duplicate) for standalone readability ---
from pathlib import Path
from .parser import read_properties, find_skill_md
from typing import List, Dict, Any


# -------------------------------------------------------------------------
# Metadata Reading
# -------------------------------------------------------------------------

def read_all_skills_metadata(skills_folder: str) -> List[Dict[str, Any]]:
    """
    Read metadata from every SKILL.md file in a directory and return as dicts.

    Iterates over sorted subdirectories of ``skills_folder``, reads each
    skill's properties (name, description, license, compatibility, allowed
    tools, metadata, body), and returns them as a list of dictionaries.

    This is the primary data-loading function used by the benchmark scripts
    and the ``SkillAwareAgent`` to discover available skills.

    Args:
        skills_folder: Path to the folder containing skill subdirectories.

    Returns:
        A list of dicts, each containing keys:
            name, description, license, compatibility, allowed_tools,
            metadata, body, path.
        Returns an empty list if the folder does not exist.
    """
    skills_path = Path(skills_folder).resolve()
    
    if not skills_path.exists():
        print(f"❌ Folder not found: {skills_path}")
        return []
    
    skills_metadata = []
    skill_dirs = []
    
    # Discover all subdirectories that contain a valid SKILL.md file
    for item in sorted(skills_path.iterdir()):
        if item.is_dir() and find_skill_md(item):
            skill_dirs.append(item)
    
    # Parse each skill's SKILL.md into a metadata dictionary
    for skill_dir in skill_dirs:
        try:
            props = read_properties(skill_dir)
            
            # Convert the parsed properties object into a plain dictionary
            skill_data = {
                'name': props.name,
                'description': props.description,
                'license': props.license,
                'compatibility': props.compatibility,
                'allowed_tools': props.allowed_tools,
                'metadata': props.metadata,
                'body': props.body,         # The full Markdown body (skill instructions)
                'path': str(skill_dir)      # Absolute path for later reference
            }
            
            skills_metadata.append(skill_data)
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to read {skill_dir.name}: {e}")
            continue
    
    return skills_metadata


# -------------------------------------------------------------------------
# Display Utilities
# -------------------------------------------------------------------------

def display_skills_metadata(skills_metadata: List[Dict[str, Any]]) -> None:
    """
    Pretty-print a detailed summary of all loaded skills.

    Renders each skill as a box-drawing card showing its description,
    license, compatibility, allowed tools, metadata, a body preview
    (first 80 chars), and the filesystem path.

    Args:
        skills_metadata: List of skill metadata dicts (as returned by
                         ``read_all_skills_metadata``).
    """
    if not skills_metadata:
        print("\n⚠️  No skills found.\n")
        return
    
    # Top banner
    print("\n" + "█" * 100)
    print("█" + " " * 98 + "█")
    print("█" + f" SKILLS METADATA SUMMARY ({len(skills_metadata)} skills found)".center(98) + "█")
    print("█" + " " * 98 + "█")
    print("█" * 100)
    
    for idx, skill in enumerate(skills_metadata, 1):
        # --- Card header: skill index and name ---
        skill_name = f"[{idx}] {skill['name']}"
        print(f"\n┌─ {skill_name}")
        print("│")
        
        # Description — split into lines of 85 chars for readability
        desc = skill['description']
        if len(desc) > 90:
            desc = desc[:87] + "..."
        print(f"│  📝 Description")
        for line in [desc[i:i+85] for i in range(0, len(desc), 85)]:
            print(f"│     {line}")
        
        # License
        if skill.get('license'):
            print(f"│  ⚖️  License:           {skill['license']}")
        
        # Compatibility
        if skill.get('compatibility'):
            compat = skill['compatibility']
            if len(compat) > 70:
                compat = compat[:67] + "..."
            print(f"│  🔧 Compatibility:      {compat}")
        
        # Allowed Tools
        if skill.get('allowed_tools'):
            tools = skill['allowed_tools']
            if len(tools) > 70:
                tools = tools[:67] + "..."
            print(f"│  🛠️  Allowed Tools:     {tools}")
        
        # Metadata
        if skill.get('metadata'):
            meta_str = str(skill['metadata'])
            if len(meta_str) > 70:
                meta_str = meta_str[:67] + "..."
            print(f"│  📦 Metadata:           {meta_str}")
        
        # Body — show a brief preview (first 80 chars, newlines collapsed)
        if skill.get('body'):
            body = skill['body'].strip()
            body_preview = body[:80].replace('\n', ' ')
            if len(body) > 80:
                body_preview = body_preview[:77] + "..."
            print(f"│  📄 Body:")
            print(f"│     {body_preview}")
        
        # Filesystem path — truncated from the left if too long
        path = skill['path']
        if len(path) > 85:
            path = "..." + path[-82:]
        print(f"│  📁 Path:               {path}")
        print("└" + "─" * 98)
    
    print("█" * 100 + "\n")


# -------------------------------------------------------------------------
# Response Parsing Utilities
# -------------------------------------------------------------------------

def parse_skill_from_response(response: str, skills_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Match skill names mentioned in a free-text agent response to loaded metadata.

    Performs a case-insensitive substring search: if a skill's name appears
    anywhere in the response, that skill is included in the results.

    Args:
        response: The raw text response from the agent.
        skills_metadata: List of skill metadata dicts to search through.

    Returns:
        A list of matching skill metadata dicts (may be empty).
    """
    # Treat the entire response as a potential container of skill names
    skill_name = response.strip()
    skills = []

    # Check each known skill name against the response (case-insensitive)
    for skill_meta in skills_metadata:
        if skill_meta['name'].lower() in skill_name.lower():
            skills.append(skill_meta)
    
    return skills

import json


def parse_skills_from_json_response(json_response: str, skills_hub_dir: str) -> List[Dict[str, Any]]:
    """
    Parse skill names from a structured JSON response and look them up in the hub.

    Expects the JSON to contain a "skills" (or "Skills") key with a list of
    skill name strings. Each name is matched case-insensitively against
    the skills available in ``skills_hub_dir``.

    This is the main parsing function used in the benchmark pipeline after
    the skill-selection agent returns its JSON-formatted response.

    Args:
        json_response: A JSON string, e.g. '{"skills": ["skill-a", "skill-b"]}'.
        skills_hub_dir: Path to the directory containing all skill subdirectories.

    Returns:
        A list of matched skill metadata dicts. Returns an empty list if
        JSON parsing fails or no skills match.
    """
    # Load all available skills from the hub for matching
    skills_hub = read_all_skills_metadata(Path(skills_hub_dir).resolve())
    
    try:
        data = json.loads(json_response)

        # Support both "skills" and "Skills" key casing
        skill_names = data.get("skills", []) or data.get("Skills", [])
        
        if not skill_names:
            return []
        
        # Build a case-insensitive lookup dict: lowered name -> full metadata
        skills_lookup = {skill['name'].lower(): skill for skill in skills_hub}
        
        # Match each requested skill name against the available skills
        matched_skills = [
            skills_lookup[skill_name.lower()]
            for skill_name in skill_names
            if skill_name.lower() in skills_lookup.keys()
        ]
        
        return matched_skills
        
    except (json.JSONDecodeError, KeyError, TypeError):
        # If the response is not valid JSON or missing expected keys, return empty
        return []

def parse_message_from_json_response(json_response: str) -> str:
    """
    Extract a message string from a JSON agent response.

    Tries several common field names in order of priority:
    "message", "Message", "reasoning", "response". If JSON parsing fails
    entirely, falls back to returning the raw input string (stripped).

    Args:
        json_response: The raw JSON string from the agent.

    Returns:
        The extracted message text, or the raw input if parsing fails.
    """
    try:
        data = json.loads(json_response)

        # Try multiple common field names (first non-empty wins)
        message = (
            data.get("message", "") or
            data.get("Message", "") or
            data.get("reasoning", "") or
            data.get("response", "")
        )
        return str(message).strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Graceful fallback: return the raw string itself
        return json_response.strip()
    
# -------------------------------------------------------------------------
# Sampling & Text Utilities
# -------------------------------------------------------------------------

def get_random_skills(skills_metadata: List[Dict[str, Any]], num_skills: int) -> List[Dict[str, Any]]:
    """
    Randomly sample a subset of skills from the given metadata list.

    If ``num_skills`` is greater than or equal to the total available,
    the full list is returned (no sampling needed).

    Args:
        skills_metadata: Full list of skill metadata dicts.
        num_skills: Number of skills to randomly select.

    Returns:
        A list of ``num_skills`` randomly chosen skill metadata dicts.
    """
    import random
    if num_skills >= len(skills_metadata):
        return skills_metadata
    return random.sample(skills_metadata, num_skills)


import re


def replace_skills(text, replacement="capability"):
    """
    Replace all occurrences of "skill" / "skills" in text with a custom term.

    Automatically handles pluralization: if the replacement word ends with
    'y', the plural form uses '-ies' (e.g., "capability" -> "capabilities");
    otherwise it simply appends 's'.

    Uses word-boundary regex (\b) for case-insensitive whole-word matching
    to avoid replacing substrings inside other words.

    Args:
        text: The input string to process.
        replacement: The singular replacement word (default: "capability").

    Returns:
        The text with all "skill"/"skills" replaced by the given term.
    """
    # Build the correct plural form of the replacement word
    if replacement.endswith('y'):
        plural = replacement[:-1] + 'ies'   # e.g., capability -> capabilities
    else:
        plural = replacement + 's'          # e.g., tool -> tools
    
    # Replace singular "skill" first, then plural "skills"
    text = re.sub(r"\bskill\b", replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\bskills?\b", plural, text, flags=re.IGNORECASE)
    return text
