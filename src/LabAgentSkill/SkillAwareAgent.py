"""
Skill-Aware Agent Module

This module provides a LangChain-based agent with optional chat history support
and integrated skill loading from SKILL.md files.

Features:
    - Multi-turn conversation with optional chat history
    - Dynamic skill loading and context enrichment
    - Configurable history tracking per call
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import re

from openai import BadRequestError
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.messages import RemoveMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from jinja2 import Environment, FileSystemLoader
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
import uuid
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
USE_CHAT_HISTORY = True  # Default: enable chat history


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Keep only the last few messages to fit the context window.

    Strategy:
        - Always keep the first message (usually system prompt)
        - Keep the last 3-4 messages depending on parity
    """
    messages = state["messages"]

    if len(messages) <= 3:
        return None

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class SkillAwareAgent:
    """
    A LangChain agent with skill awareness and optional chat history support.
    
    Attributes:
        agent: The underlying LangChain agent
        llm: The LLM instance (gpt-4o-mini by default)
        chat_history: The chat history storage
        system_prompt: The system prompt for the agent
        use_chat_history: Global flag for chat history usage
    """
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant that can use skills to solve tasks.",
        model: str = "gpt-4o-mini",
        use_chat_history: bool = True,
        use_trim_messages: bool = True,
        thread_id: str = "1",
        temperature: float = None,
        max_tokens: int = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Skill-Aware Agent.
        
        Args:
            system_prompt: The system prompt to use for the agent
            model: The LLM model to use (default: gpt-4o-mini)
            use_chat_history: Whether to enable chat history by default
            use_trim_messages: Whether to enable trim-messages middleware
            thread_id: Thread ID for checkpointed conversation state
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens in response
            base_url: Base URL for the API endpoint (e.g. "http://localhost:8000/v1"
                      for a local vLLM server). If None, uses the default OpenAI endpoint.
            api_key: API key override. For local vLLM you can pass "EMPTY" or any
                     dummy string. If None, falls back to OPENAI_API_KEY env var.
        """
        self.use_chat_history = use_chat_history
        self.use_trim_messages = use_trim_messages
        self.thread_id = thread_id
        self.system_prompt = system_prompt
        self.chat_history = InMemoryChatMessageHistory()
        self.runtime_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        self.model = model
        
        # Initialize the LLM — build kwargs dynamically so that only
        # explicitly provided parameters are forwarded to ChatOpenAI.
        llm_kwargs: Dict[str, Any] = {"model": model}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        if base_url is not None:
            llm_kwargs["base_url"] = base_url
        if api_key is not None:
            llm_kwargs["api_key"] = api_key

        self.llm = ChatOpenAI(**llm_kwargs)
        
        middleware = [trim_messages] if use_trim_messages else []
        
        # Create the agent
        self.agent = create_agent(
            self.llm,
            system_prompt=system_prompt,
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )
        
        print(f"✓ SkillAwareAgent initialized")
        print(f"  Model: {model}")
        if base_url:
            print(f"  Base URL: {base_url}")
        print(f"  Chat History: {'ENABLED ✓' if use_chat_history else 'DISABLED ✗'}")
        print(f"  Trim Messages: {'ENABLED ✓' if use_trim_messages else 'DISABLED ✗'}")
    
    def chat(
        self,
        user_input: str,
        custom_system_prompt: Optional[str] = None,
        use_history: Optional[bool] = None,
        config: Optional[RunnableConfig] = None
    ) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            custom_system_prompt: Optional custom system prompt for this call
            user_input: The user's input message
            use_history: Override to enable/disable history for this call
                        If None, uses self.use_chat_history setting
            config: Optional RunnableConfig (thread_id, etc.)
        
        Returns:
            The agent's response string
        """
        # Determine if we should use history for this call
        should_use_history = use_history if use_history is not None else self.use_chat_history
        
        # Use runtime config (for checkpointed state)
        runtime_config = config if config is not None else self.runtime_config
        
        # Invoke agent (checkpointer automatically manages history if enabled)
        if custom_system_prompt is not None:
            system_message = custom_system_prompt
        else:
            system_message = self.system_prompt
        
        # Retry loop: on context-length BadRequestError, halve user_input and retry
        MAX_TRUNCATION_RETRIES = 5
        current_input = user_input
        current_system_message = system_message 
        for attempt in range(MAX_TRUNCATION_RETRIES + 1):
            try:
                if "google" in self.model.lower():
                    result = self.agent.invoke(
                        {"messages": [HumanMessage(content=current_system_message + current_input)]},
                        runtime_config,
                    )
                else:
                    result = self.agent.invoke(
                        {"messages": [SystemMessage(content=current_system_message), HumanMessage(content=current_input)]},
                        runtime_config,
                    )
                break  # success
            except BadRequestError as e:
                err_msg = str(e).lower()
                if "context length" in err_msg or "maximum" in err_msg and "token" in err_msg:
                    if attempt < MAX_TRUNCATION_RETRIES:
                        old_len = len(current_input)
                        current_input = current_input[: old_len // 2]
                        runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())  # Clear history to free up context space
                        # runtime_config["middleware"] = [REMOVE_ALL_MESSAGES()]  # Clear messages in the graph
                    else:
                        print("✗ Context length still exceeded after max retries. Re-raising.")
                        raise
                else:
                    raise  # unrelated BadRequestError, don't swallow it

        # Extract response
        if isinstance(result, dict):
            if 'messages' in result:
                # Get the last message from the result
                last_msg = result['messages'][-1]
                if isinstance(last_msg, AIMessage):
                    response = last_msg.content
                else:
                    response = str(last_msg)
            else:
                response = result.get('output', str(result))
        else:
            response = str(result)
        
        # Update manual chat history for display purposes
        if should_use_history:
            self.chat_history.add_user_message(user_input)
            self.chat_history.add_ai_message(response)
        
        return response
    
    def clear_history(self) -> None:
        """Clear the default chat history."""
        self.chat_history.clear()
        # print("✓ Chat history cleared")
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the default chat history.
        
        Returns:
            Dict with total, user_messages, agent_messages, and turns
        """
        messages = self.chat_history.messages
        user_msgs = sum(1 for m in messages if isinstance(m, HumanMessage))
        agent_msgs = sum(1 for m in messages if isinstance(m, AIMessage))
        
        return {
            'total': len(messages),
            'user_messages': user_msgs,
            'agent_messages': agent_msgs,
            'turns': user_msgs
        }
    
    def display_history(self) -> None:
        """Display the default chat history in a readable format."""
        print("\n" + "="*80)
        print(f"CHAT HISTORY ({len(self.chat_history.messages)} messages)")
        print("="*80)
        
        for i, msg in enumerate(self.chat_history.messages, 1):
            role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 Agent"
            content = msg.content
            print(f"\n[{i}] {role}:")
            print(f"{content}")
            print("-"*80)
    
    def display_human_and_ai_message_history(self, max_width: int = 78) -> None:
        """
        Display chat history with alternating human and AI messages in an easy-to-read format.
        
        Shows each conversation turn with clear visual separation between user and agent messages.
        
        Args:
            max_width: Maximum width for text wrapping (default: 78)
        """
        messages = self.chat_history.messages
        
        if not messages:
            print("\n⚠️  No messages in chat history")
            return
        
        total_msgs = len(messages)
        total_turns = sum(1 for m in messages if isinstance(m, HumanMessage))
        
        # Header
        print("\n" + "╔" + "═"*78 + "╗")
        print(f"║ {'CONVERSATION HISTORY':<76} ║")
        print(f"║ {f'Total Messages: {total_msgs} | Turns: {total_turns}':<76} ║")
        print("╚" + "═"*78 + "╝")
        
        turn_num = 0
        i = 0
        
        while i < len(messages):
            # Get human message
            if i < len(messages) and isinstance(messages[i], HumanMessage):
                turn_num += 1
                
                # Turn separator
                print(f"\n┌─ TURN {turn_num} " + "─"*(73 - len(str(turn_num))))
                
                # User message
                print("│")
                print("│ 👤 USER:")
                user_text = messages[i].content
                user_lines = user_text.split('\n')
                for line in user_lines:
                    # Wrap long lines
                    if len(line) > max_width:
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 <= max_width:
                                current_line += word + " "
                            else:
                                if current_line:
                                    print(f"│   {current_line.strip()}")
                                current_line = word + " "
                        if current_line:
                            print(f"│   {current_line.strip()}")
                    else:
                        print(f"│   {line}")
                i += 1
                
            # Get corresponding AI message
            if i < len(messages) and isinstance(messages[i], AIMessage):
                print("│")
                print("│ 🤖 AGENT:")
                
                # Extract content safely
                agent_content = messages[i].content
                
                # Ensure it's a string (in case it's wrapped in an object)
                if isinstance(agent_content, dict):
                    # If content is a dict, try to extract text
                    if 'messages' in agent_content:
                        agent_text = str(agent_content.get('messages', agent_content))
                    else:
                        agent_text = str(agent_content)
                elif isinstance(agent_content, str):
                    agent_text = agent_content
                else:
                    agent_text = str(agent_content)
                
                # Print the content line by line
                agent_lines = agent_text.split('\n')
                for line in agent_lines:
                    # Wrap long lines
                    if len(line) > max_width:
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line) + len(word) + 1 <= max_width:
                                current_line += word + " "
                            else:
                                if current_line:
                                    print(f"│   {current_line.strip()}")
                                current_line = word + " "
                        if current_line:
                            print(f"│   {current_line.strip()}")
                    else:
                        print(f"│   {line}")
                print("│")
                print("└" + "─"*78)
                i += 1
            else:
                print("│")
                print("└" + "─"*78)

        # Footer with summary
        print(f"\n╔" + "═"*78 + "╗")
        print(f"║ Summary: {total_turns} turn(s) | {total_msgs} message(s) {' '*42} ║")
        print("╚" + "═"*78 + "╝\n")
    
    def get_human_ai_message_history(self) -> List[Dict[str, str]]:
        """
        Return chat history as a list of dicts with alternating human and AI messages.
        
        Each dict contains:
            - role: "human" or "ai"
            - content: The message content string
            - turn: The conversation turn number (1-indexed)
        
        Returns:
            List of message dicts, e.g.:
            [
                {"role": "human", "content": "...", "turn": 1},
                {"role": "ai",    "content": "...", "turn": 1},
                {"role": "human", "content": "...", "turn": 2},
                ...
            ]
        """
        messages = self.chat_history.messages
        history = []
        turn = 0

        for msg in messages:
            if isinstance(msg, HumanMessage):
                turn += 1
                history.append({
                    "role": "human",
                    "content": msg.content,
                    "turn": turn,
                })
            elif isinstance(msg, AIMessage):
                # Extract content safely
                content = msg.content
                if isinstance(content, dict):
                    content = str(content.get("messages", content))
                elif not isinstance(content, str):
                    content = str(content)
                history.append({
                    "role": "ai",
                    "content": content,
                    "turn": turn,
                })

        return history
    
    def export_conversation_to_text(self, filepath: Path) -> None:
        """
        Export the chat history to a text file.
        
        Args:
            filepath: Path where to save the conversation
        """
        messages = self.chat_history.messages
        
        if not messages:
            print("⚠️  No messages to export")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"CONVERSATION HISTORY ({len(messages)} messages)\n")
            f.write("="*80 + "\n\n")
            
            turn_num = 0
            for i, msg in enumerate(messages, 1):
                if isinstance(msg, HumanMessage):
                    turn_num += 1
                    f.write(f"\n[Turn {turn_num}] USER:\n")
                    f.write(f"{msg.content}\n")
                else:
                    f.write(f"\n[Turn {turn_num}] AGENT:\n")
                    f.write(f"{msg.content}\n")
                f.write("-"*80 + "\n")
        
        print(f"✓ Conversation exported to {filepath}")
        


# ============================================================================
# SKILL LOOP FUNCTIONS
# ============================================================================

def extract_required_skills(agent_response: str) -> List[str]:
    """
    Extract skill names from agent's response.
    
    Looks for patterns like [SKILLS NEEDED: skill1, skill2] or mentions of skill names.
    
    Args:
        agent_response: The response from the agent
    
    Returns:
        List of skill names
    """
    # Pattern 1: [SKILLS NEEDED: ...]
    match = re.search(r'\[SKILLS NEEDED:\s*([^\]]+)\]', agent_response, re.IGNORECASE)
    if match:
        skills_str = match.group(1)
        skills = [s.strip() for s in skills_str.split(',')]
        return skills
    
    # Pattern 2: I need the ... skill
    matches = re.findall(
        r'(?:need|require|use).*?the\s+([a-z\-]+)\s+skill',
        agent_response,
        re.IGNORECASE
    )
    return list(set(matches))


def load_skill_content(skill_name: str, skills_folder: Path) -> Optional[Dict[str, Any]]:
    """
    Load the full content of a skill by name.
    
    Args:
        skill_name: Name of the skill to load
        skills_folder: Path to the skills folder
    
    Returns:
        Dict with skill metadata or None if not found
    """
    from LabAgentSkill import skills_utils
    
    skill_dir = skills_folder / skill_name
    
    if not skill_dir.exists():
        return None
    
    try:
        props = skills_utils.read_all_skills_metadata(skills_folder)
        for skill in props:
            if skill['name'] == skill_name:
                return skill
    except Exception as e:
        print(f"Warning: Failed to load skill {skill_name}: {e}")
    
    return None


def build_enriched_prompt(
    task: str,
    required_skills: List[str],
    skills_folder: Path
) -> str:
    """
    Build an enriched prompt with skill content integrated.
    
    Args:
        task: The main task description
        required_skills: List of skill names to include
        skills_folder: Path to the skills folder
    
    Returns:
        The enriched prompt string
    """
    prompt = f"Task: {task}\n\n"
    
    if required_skills:
        prompt += "="*80 + "\n"
        prompt += "RELEVANT SKILLS CONTEXT:\n"
        prompt += "="*80 + "\n"
        
        for skill_name in required_skills:
            skill = load_skill_content(skill_name, skills_folder)
            if skill:
                prompt += f"\n[SKILL: {skill['name']}]\n"
                prompt += f"Description: {skill['description']}\n"
                if skill.get('body'):
                    prompt += f"Content:\n{skill['body']}\n"
                prompt += "-"*80 + "\n"
    
    return prompt


def skill_loop_with_history(
    agent: SkillAwareAgent,
    task: str,
    skills_folder: Path,
    use_history: Optional[bool] = None,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Execute skill loop with optional chat history support.
    
    Process:
        1. Agent declares which skills it needs
        2. Extract skill names from agent response
        3. Load skill content from filesystem
        4. Build enriched prompt with skill context
        5. Agent solves task with enriched context
    
    Args:
        agent: The SkillAwareAgent instance
        task: The task to execute
        skills_folder: Path to skills folder
        use_history: Override to enable/disable history (optional)
        config: Optional RunnableConfig (thread_id, etc.)
    
    Returns:
        Dict with required_skills, loaded_skills, and response
    """
    # Determine if we should use history
    should_use_history = (
        use_history if use_history is not None else agent.use_chat_history
    )

    runtime_config = config if config is not None else agent.runtime_config
    
    # Step 1: Ask agent what skills are needed
    skill_declaration_task = f"""Analyze this task and declare which skills you need:

Task: {task}

Respond with: [SKILLS NEEDED: skill1, skill2, ...]"""
    
    skill_result = agent.agent.invoke(
        {"messages": [HumanMessage(content=skill_declaration_task)]}, 
        runtime_config
    )
    
    # Extract skill output
    if isinstance(skill_result, dict) and 'messages' in skill_result:
        last_msg = skill_result['messages'][-1]
        skill_output = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    else:
        skill_output = skill_result.get('output', str(skill_result))
    
    # Step 2: Extract and parse skills
    required_skills = extract_required_skills(skill_output)
    
    # Step 3: Load skill content
    loaded_skills = []
    for skill_name in required_skills:
        skill = load_skill_content(skill_name, skills_folder)
        if skill:
            loaded_skills.append(skill)
    
    # Step 4: Build enriched prompt and solve
    enriched_task = build_enriched_prompt(task, required_skills, skills_folder)
    
    # Invoke agent with enriched task
    result = agent.agent.invoke(
        {"messages": [HumanMessage(content=enriched_task)]}, 
        runtime_config
    )
    
    # Extract final output
    if isinstance(result, dict) and 'messages' in result:
        last_msg = result['messages'][-1]
        final_output = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    else:
        final_output = result.get('output', str(result))
    
    # Update manual chat history for display purposes
    if should_use_history:
        agent.chat_history.add_user_message(task)
        agent.chat_history.add_ai_message(final_output)
    
    return {
        'required_skills': required_skills,
        'loaded_skills': loaded_skills,
        'response': final_output
    }


# ============================================================================
# SYSTEM PROMPT LOADING
# ============================================================================

def load_system_prompt(
    prompts_folder: Path,
    template_name: str = "system_prompt_template.jinja",
    skill_context: str = ""
) -> str:
    """
    Load and render the system prompt template.
    
    Args:
        prompts_folder: Path to the prompts folder
        template_name: Name of the template file
        skill_context: Context string with available skills
    
    Returns:
        The rendered system prompt
    """
    env = Environment(loader=FileSystemLoader(str(prompts_folder)))
    template = env.get_template(template_name)
    system_prompt = template.render(SKILL_CONTEXT=skill_context)
    return system_prompt


def load_system_prompt_with_skills(
    prompts_folder: Path,
    skills_folder: Path,
    template_name: str = "system_prompt_template.jinja"
) -> str:
    """
    Load system prompt template and automatically include available skills.
    
    Args:
        prompts_folder: Path to the prompts folder
        skills_folder: Path to the skills folder
        template_name: Name of the template file
    
    Returns:
        The rendered system prompt with skill context
    """
    from LabAgentSkill import skills_utils
    
    # Load all skills
    all_skills = skills_utils.read_all_skills_metadata(skills_folder)
    
    # Build skill context
    skill_context = "\n".join([
        f"- **{skill['name']}**: {skill['description']} "
        f"(Path: {skills_folder / skill['name'] / 'SKILL.md'})"
        for skill in all_skills
    ])
    
    # Load and render template
    return load_system_prompt(
        prompts_folder,
        template_name,
        skill_context
    )
