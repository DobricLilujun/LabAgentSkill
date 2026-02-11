from pathlib import Path
from .parser import read_properties, find_skill_md

def list_skills_in_folder(skills_folder: str) -> None:
    """
    Print all skills in the target folder with their names and descriptions.
    
    Args:
        skills_folder: Path to the folder containing skill subdirectories
    """
    skills_path = Path(skills_folder).resolve()
    
    if not skills_path.exists():
        print(f"❌ Folder not found: {skills_path}")
        return
    
    # Get all skill directories (those containing SKILL.md)
    skill_dirs = []
    for item in sorted(skills_path.iterdir()):
        if item.is_dir() and find_skill_md(item):
            skill_dirs.append(item)
    
    if not skill_dirs:
        print(f"ℹ️  No skills found in: {skills_path}")
        return
    
    print("\n" + "=" * 80)
    print(f"SKILLS IN: {skills_path.name}")
    print("=" * 80)
    print(f"Total: {len(skill_dirs)} skills\n")
    
    for idx, skill_dir in enumerate(skill_dirs, 1):
        try:
            props = read_properties(skill_dir)
            name = props.name
            description = props.description
            
            # Truncate long descriptions
            if len(description) > 70:
                description = description[:67] + "..."
            
            print(f"[{idx}] {name}")
            print(f"    └─ {description}")
            
        except Exception as e:
            print(f"[{idx}] ❌ Error reading {skill_dir.name}: {e}")
    
    print("\n" + "=" * 80 + "\n")


from pathlib import Path
from .parser import read_properties, find_skill_md
from typing import List, Dict, Any

def read_all_skills_metadata(skills_folder: str) -> List[Dict[str, Any]]:
    """
    Read all skill metadata from SKILL.md files in a directory.
    
    Args:
        skills_folder: Path to the folder containing skill subdirectories
        
    Returns:
        List of dictionaries containing skill metadata
    """
    skills_path = Path(skills_folder).resolve()
    
    if not skills_path.exists():
        print(f"❌ Folder not found: {skills_path}")
        return []
    
    skills_metadata = []
    skill_dirs = []
    
    # Find all directories containing SKILL.md
    for item in sorted(skills_path.iterdir()):
        if item.is_dir() and find_skill_md(item):
            skill_dirs.append(item)
    
    # Read metadata from each skill
    for skill_dir in skill_dirs:
        try:
            props = read_properties(skill_dir)
            
            # Convert to dictionary
            skill_data = {
                'name': props.name,
                'description': props.description,
                'license': props.license,
                'compatibility': props.compatibility,
                'allowed_tools': props.allowed_tools,
                'metadata': props.metadata,
                'body': props.body,
                'path': str(skill_dir)
            }
            
            skills_metadata.append(skill_data)
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to read {skill_dir.name}: {e}")
            continue
    
    return skills_metadata


def display_skills_metadata(skills_metadata: List[Dict[str, Any]]) -> None:
    """
    Pretty print skills metadata in a formatted table.
    
    Args:
        skills_metadata: List of skill metadata dictionaries
    """
    if not skills_metadata:
        print("\n⚠️  No skills found.\n")
        return
    
    print("\n" + "█" * 100)
    print("█" + " " * 98 + "█")
    print("█" + f" SKILLS METADATA SUMMARY ({len(skills_metadata)} skills found)".center(98) + "█")
    print("█" + " " * 98 + "█")
    print("█" * 100)
    
    for idx, skill in enumerate(skills_metadata, 1):
        # Skill name and index
        skill_name = f"[{idx}] {skill['name']}"
        print(f"\n┌─ {skill_name}")
        print("│")
        
        # Description
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
        
        # Body preview
        if skill.get('body'):
            body = skill['body'].strip()
            body_preview = body[:80].replace('\n', ' ')
            if len(body) > 80:
                body_preview = body_preview[:77] + "..."
            print(f"│  📄 Body:")
            print(f"│     {body_preview}")
        
        # Path
        path = skill['path']
        if len(path) > 85:
            path = "..." + path[-82:]
        print(f"│  📁 Path:               {path}")
        print("└" + "─" * 98)
    
    print("█" * 100 + "\n")


def parse_skill_from_response(response: str, skills_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse the skill name from the agent's response and return the corresponding skill metadata.
    
    Args:
        response: The raw response string from the agent
        skills_metadata: List of skill metadata dictionaries

    Returns:
        The metadata dictionary of the identified skill, or None if not found
    """
    # Extract skill name from response (assuming response is just the skill name)
    skill_name = response.strip()
    skills = []
    # Search for the skill in metadata
    for skill_meta in skills_metadata:
        if skill_meta['name'].lower() in skill_name.lower():
            skills.append(skill_meta)
    
    return skills

import json

def parse_skills_from_json_response(json_response: str, skills_hub_dir: str) -> List[Dict[str, Any]]:
    """
    Parse skill names from a JSON response and return the corresponding skill metadata.
    
    Args:
        json_response: The raw JSON response string from the agent
        skills_hub_dir: Directory containing the skills hub

    Returns:
        List of skill metadata dictionaries. Returns empty list if parsing fails
        or no skills are found.
    """
    skills_hub = read_all_skills_metadata(Path(skills_hub_dir).resolve())
    
    try:
        data = json.loads(json_response)
        skill_names = data.get("skills", []) or data.get("Skills", [])
        
        if not skill_names:
            return []
        
        # Create a lookup dictionary for efficient matching
        skills_lookup = {skill['name'].lower(): skill for skill in skills_hub}
        
        # Match requested skills with available skills
        matched_skills = [
            skills_lookup[skill_name.lower()]
            for skill_name in skill_names
            if skill_name.lower() in skills_lookup
        ]
        
        return matched_skills
        
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

def parse_message_from_json_response(json_response: str) -> str:
    """
    Parse message from a JSON response.
    
    Args:
        json_response: The raw JSON response string from the agent

    Returns:
        The extracted message string. Returns empty string if parsing fails.
    """
    try:
        data = json.loads(json_response)
        # Try common message field names
        message = (
            data.get("message", "") or
            data.get("Message", "") or
            data.get("reasoning", "") or
            data.get("response", "")
        )
        return str(message).strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return json_response.strip()