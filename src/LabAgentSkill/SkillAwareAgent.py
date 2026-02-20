"""
Skill-Aware Agent Module

This module provides a LangChain/LangGraph-based conversational agent with:
    - Multi-turn conversation with optional, configurable chat history
    - Automatic message trimming middleware to stay within LLM context windows
    - Dynamic skill loading and context enrichment from SKILL.md files
    - Support for multiple LLM backends (OpenAI, Anthropic Claude, local vLLM)
    - Automatic retry with input truncation on context-length errors
    - Jinja2-based system prompt templating with skill injection

Typical usage flow:
    1. Instantiate ``SkillAwareAgent`` with desired model and settings
    2. Call ``agent.chat(user_input, custom_system_prompt)`` for each turn
    3. Optionally use ``skill_loop_with_history()`` for the full
       skill-selection → skill-loading → enriched-execution pipeline
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
from langchain_anthropic import ChatAnthropic
# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Default global flag — individual agents can override this in their constructor.
USE_CHAT_HISTORY = True


# ----------------------------------------------------------------------------
# Middleware: Automatic Message Trimming
# ----------------------------------------------------------------------------
# This middleware runs *before* the LLM call and trims the message list to
# prevent context-window overflow. It is registered via the @before_model
# decorator and can be optionally attached to an agent at construction time.

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Trim the conversation to fit the LLM's context window.

    Strategy:
        - If there are 3 or fewer messages, do nothing (already short enough).
        - Always preserve the **first message** (typically the system prompt).
        - Keep the **last 3 or 4 messages** (depending on parity) so that
          the most recent user–assistant exchange is always included.
        - All other intermediate messages are removed via REMOVE_ALL_MESSAGES
          followed by re-insertion of the kept messages.

    Args:
        state: The current agent state containing the full message list.
        runtime: The LangGraph runtime context.

    Returns:
        A dict with the trimmed ``messages`` list, or ``None`` if no
        trimming was necessary.
    """
    messages = state["messages"]

    # No trimming needed for very short conversations
    if len(messages) <= 3:
        return None

    # Always keep the first message (system prompt / initial context)
    first_msg = messages[0]

    # Keep the last 3 messages if total count is even, otherwise last 4,
    # so that the kept slice always starts with a user message.
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    # Replace the entire message list: remove all, then re-add the kept ones
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


# ============================================================================
# CORE AGENT CLASS
# ============================================================================

class SkillAwareAgent:
    """
    A LangChain/LangGraph agent with skill awareness and optional chat history.

    This class wraps a LangGraph ``create_agent`` graph and adds:
      - Configurable LLM backend (OpenAI, Anthropic, local vLLM)
      - Optional message-trimming middleware to manage context length
      - An in-memory chat history for display / export (separate from the
        graph's own checkpointed state)
      - Automatic retry with input truncation on context-length errors

    Attributes:
        agent: The underlying LangGraph agent graph.
        llm: The LLM instance (ChatOpenAI or ChatAnthropic).
        chat_history: In-memory message history for display / export.
        system_prompt: The default system prompt for the agent.
        use_chat_history: Whether chat history tracking is enabled.
        use_trim_messages: Whether the trimming middleware is active.
        model: The model name string.
        thread_id: Thread ID used for the LangGraph checkpointer.
        runtime_config: The default RunnableConfig passed to agent.invoke().
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

        # Separate in-memory history for display/export (not used by the LangGraph
        # checkpointer — that maintains its own state via InMemorySaver).
        self.chat_history = InMemoryChatMessageHistory()

        # Default runtime config passed to agent.invoke(); the thread_id ties
        # into the InMemorySaver checkpointer for multi-turn state management.
        self.runtime_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        self.model = model
        
        # --- Initialize the LLM ---
        # Build kwargs dynamically so that only explicitly provided parameters
        # are forwarded, avoiding sending None values to the LLM constructor.
        llm_kwargs: Dict[str, Any] = {"model": model}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        if base_url is not None:
            llm_kwargs["base_url"] = base_url
        if api_key is not None:
            llm_kwargs["api_key"] = api_key

        # Route to the appropriate LLM provider based on model name
        if "claude"in model.lower():
            self.llm = ChatAnthropic(**llm_kwargs)
        else:
            self.llm = ChatOpenAI(**llm_kwargs)

        
        # Attach the trimming middleware only if requested
        middleware = [trim_messages] if use_trim_messages else []
        
        # Create the LangGraph agent with an in-memory checkpointer
        # that persists conversation state across .invoke() calls
        # within the same thread_id.
        self.agent = create_agent(
            self.llm,
            system_prompt=system_prompt,
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )
        
        # Print initialization summary for debugging visibility
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
        # Determine if we should track history for this particular call.
        # Per-call override takes precedence over the instance-level default.
        should_use_history = use_history if use_history is not None else self.use_chat_history
        
        # Use the caller-provided config or fall back to the instance default
        runtime_config = config if config is not None else self.runtime_config
        
        # Resolve the system prompt: per-call override or instance default
        if custom_system_prompt is not None:
            system_message = custom_system_prompt
        else:
            system_message = self.system_prompt
        
        # For Claude models, generate a fresh thread_id per call to avoid
        # checkpointer conflicts (Anthropic API handles context differently).
        if "claude"in self.model.lower():
            runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())

        # --- Retry loop with automatic input truncation ---
        # If the LLM raises a BadRequestError due to context-length overflow,
        # halve the user input and retry up to MAX_TRUNCATION_RETRIES times.
        MAX_TRUNCATION_RETRIES = 5
        current_input = user_input
        current_system_message = system_message 
        for attempt in range(MAX_TRUNCATION_RETRIES + 1):
            try:
                # Google models don't support separate SystemMessage;
                # concatenate system + user into a single HumanMessage instead.
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
                break  # Invocation succeeded — exit the retry loop
            except BadRequestError as e:
                err_msg = str(e).lower()
                # Check if the error is related to context length / token limits
                if "context length" in err_msg or "maximum" in err_msg and "token" in err_msg:
                    if attempt < MAX_TRUNCATION_RETRIES:
                        old_len = len(current_input)
                        # Truncate user input to half its current length
                        current_input = current_input[: old_len // 2]
                        # Use a fresh thread to discard accumulated checkpointer state
                        runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())
                    else:
                        print("✗ Context length still exceeded after max retries. Re-raising.")
                        raise
                else:
                    # Unrelated BadRequestError — propagate immediately
                    raise

        # --- Extract the text response from the agent result ---
        # The result may be a dict with a 'messages' key (LangGraph style)
        # or a plain dict/string depending on the agent configuration.
        if isinstance(result, dict):
            if 'messages' in result:
                # Standard LangGraph output: grab the last message
                last_msg = result['messages'][-1]
                if isinstance(last_msg, AIMessage):
                    response = last_msg.content
                else:
                    response = str(last_msg)
            else:
                # Fallback for non-standard output formats
                response = result.get('output', str(result))
        else:
            response = str(result)
        
        # Record the exchange in the manual chat history (used for display,
        # export, and inclusion in JSONL result records — separate from the
        # LangGraph checkpointer's internal state).
        if should_use_history:
            self.chat_history.add_user_message(user_input)
            self.chat_history.add_ai_message(response)
        
        return response
    
    def clear_history(self) -> None:
        """Clear the in-memory chat history used for display and export."""
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
        """Print the full chat history to stdout in a simple, readable format."""
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
        Pretty-print the chat history with box-drawing characters.

        Each conversation turn is displayed as a visually separated block
        with the user message on top and the agent response below. Long
        lines are automatically word-wrapped to ``max_width``.

        Args:
            max_width: Maximum character width before wrapping (default: 78).
        """
        messages = self.chat_history.messages
        
        if not messages:
            print("\n⚠️  No messages in chat history")
            return
        
        total_msgs = len(messages)
        total_turns = sum(1 for m in messages if isinstance(m, HumanMessage))
        
        # Header with conversation statistics
        print("\n" + "╔" + "═"*78 + "╗")
        print(f"║ {'CONVERSATION HISTORY':<76} ║")
        print(f"║ {f'Total Messages: {total_msgs} | Turns: {total_turns}':<76} ║")
        print("╚" + "═"*78 + "╝")
        
        turn_num = 0
        i = 0
        
        while i < len(messages):
            # --- Process the user (human) message for this turn ---
            if i < len(messages) and isinstance(messages[i], HumanMessage):
                turn_num += 1
                
                # Visual turn separator
                print(f"\n┌─ TURN {turn_num} " + "─"*(73 - len(str(turn_num))))
                
                # Print user message with word-wrapping for long lines
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
                
            # --- Process the corresponding AI (agent) response ---
            if i < len(messages) and isinstance(messages[i], AIMessage):
                print("│")
                print("│ 🤖 AGENT:")
                
                # Safely extract content — handle dict, str, or other types
                agent_content = messages[i].content
                
                # Normalize content to a plain string regardless of type
                if isinstance(agent_content, dict):
                    # If content is a dict (rare), try to extract the 'messages' field
                    if 'messages' in agent_content:
                        agent_text = str(agent_content.get('messages', agent_content))
                    else:
                        agent_text = str(agent_content)
                elif isinstance(agent_content, str):
                    agent_text = agent_content
                else:
                    agent_text = str(agent_content)
                
                # Print agent response with word-wrapping for long lines
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

        # Footer with conversation summary
        print(f"\n╔" + "═"*78 + "╗")
        print(f"║ Summary: {total_turns} turn(s) | {total_msgs} message(s) {' '*42} ║")
        print("╚" + "═"*78 + "╝\n")
    
    def get_human_ai_message_history(self) -> List[Dict[str, str]]:
        """
        Return the chat history as a structured list of message dicts.

        This is the primary method used when saving conversation records
        to JSONL result files. Each dict contains the role, content, and
        the 1-indexed turn number.

        Returns:
            List of message dicts, e.g.:n            [
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
                # Increment the turn counter on each new user message
                turn += 1
                history.append({
                    "role": "human",
                    "content": msg.content,
                    "turn": turn,
                })
            elif isinstance(msg, AIMessage):
                # Safely extract content — handle dict or non-string edge cases
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
        Export the chat history to a plain-text file.

        Each message is prefixed with its turn number and role (USER / AGENT),
        separated by horizontal rules for readability.

        Args:
            filepath: Destination file path for the exported conversation.
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
# These standalone functions implement the "skill loop" pattern:
#   1. Agent declares which skills it needs for a given task
#   2. Skill names are extracted from the agent's response via regex
#   3. Skill content is loaded from the filesystem (SKILL.md files)
#   4. An enriched prompt combining the task + skill content is built
#   5. The agent solves the task with the enriched context
# ============================================================================

def extract_required_skills(agent_response: str) -> List[str]:
    """
    Parse skill names out of an agent's natural-language response.

    Supports two patterns:
        1. Explicit declaration: ``[SKILLS NEEDED: skill1, skill2, ...]``
        2. Implicit mention: phrases like "I need the X skill"

    Args:
        agent_response: The full text response from the agent.

    Returns:
        A deduplicated list of skill name strings (may be empty).
    """
    # Pattern 1: Structured bracket notation [SKILLS NEEDED: skill1, skill2, ...]
    match = re.search(r'\[SKILLS NEEDED:\s*([^\]]+)\]', agent_response, re.IGNORECASE)
    if match:
        skills_str = match.group(1)
        skills = [s.strip() for s in skills_str.split(',')]
        return skills
    
    # Pattern 2: Natural language mentions like "need the X skill" / "require the Y skill"
    matches = re.findall(
        r'(?:need|require|use).*?the\s+([a-z\-]+)\s+skill',
        agent_response,
        re.IGNORECASE
    )
    # Deduplicate while preserving order
    return list(set(matches))


def load_skill_content(skill_name: str, skills_folder: Path) -> Optional[Dict[str, Any]]:
    """
    Load the full metadata (name, description, body) for a single skill.

    Looks for a subdirectory matching ``skill_name`` inside ``skills_folder``,
    reads the SKILL.md file via ``skills_utils.read_all_skills_metadata()``,
    and returns the matching entry.

    Args:
        skill_name: The name of the skill (must match the folder name).
        skills_folder: Root directory containing all skill subdirectories.

    Returns:
        A dict with keys 'name', 'description', 'body', etc., or None
        if the skill directory does not exist or loading fails.
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
    Construct a prompt that combines the original task with relevant skill content.

    Each loaded skill's name, description, and body are appended as clearly
    delimited sections so the LLM can reference them when solving the task.

    Args:
        task: The original user task description.
        required_skills: List of skill names to include as context.
        skills_folder: Root directory containing all skill subdirectories.

    Returns:
        The enriched prompt string ready to be sent to the agent.
    """
    prompt = f"Task: {task}\n\n"
    
    # Append each loaded skill as a labeled, delimited block
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
    Execute the full skill-selection → loading → enriched-execution pipeline.

    This orchestrates the complete skill loop in five steps:
        1. Ask the agent to declare which skills it needs for the task.
        2. Parse skill names from the agent's response.
        3. Load the corresponding skill content from the filesystem.
        4. Build an enriched prompt combining the task + skill bodies.
        5. Invoke the agent with the enriched prompt to produce the final answer.

    Chat history is optionally updated with the original task and final response.

    Args:
        agent: The SkillAwareAgent instance to use.
        task: The user's task description.
        skills_folder: Path to the directory containing skill subdirectories.
        use_history: Override to enable/disable history for this call.
                     If None, uses the agent's default setting.
        config: Optional RunnableConfig (e.g., custom thread_id).

    Returns:
        Dict with keys:
            - ``required_skills``: List of skill names the agent requested.
            - ``loaded_skills``: List of skill metadata dicts that were loaded.
            - ``response``: The agent's final answer string.
    """
    # Determine if we should track history for this call
    should_use_history = (
        use_history if use_history is not None else agent.use_chat_history
    )

    runtime_config = config if config is not None else agent.runtime_config
    
    # --- Step 1: Ask the agent to declare needed skills ---
    skill_declaration_task = f"""Analyze this task and declare which skills you need:

Task: {task}

Respond with: [SKILLS NEEDED: skill1, skill2, ...]"""
    
    skill_result = agent.agent.invoke(
        {"messages": [HumanMessage(content=skill_declaration_task)]}, 
        runtime_config
    )
    
    # Extract the skill declaration text from the agent's response
    if isinstance(skill_result, dict) and 'messages' in skill_result:
        last_msg = skill_result['messages'][-1]
        skill_output = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    else:
        skill_output = skill_result.get('output', str(skill_result))
    
    # --- Step 2: Parse skill names from the agent's response ---
    required_skills = extract_required_skills(skill_output)
    
    # --- Step 3: Load skill content from the filesystem ---
    loaded_skills = []
    for skill_name in required_skills:
        skill = load_skill_content(skill_name, skills_folder)
        if skill:
            loaded_skills.append(skill)
    
    # --- Step 4: Build enriched prompt and invoke the agent ---
    enriched_task = build_enriched_prompt(task, required_skills, skills_folder)
    
    # --- Step 5: Invoke the agent with skill-enriched context ---
    result = agent.agent.invoke(
        {"messages": [HumanMessage(content=enriched_task)]}, 
        runtime_config
    )
    
    # Extract the final answer text
    if isinstance(result, dict) and 'messages' in result:
        last_msg = result['messages'][-1]
        final_output = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    else:
        final_output = result.get('output', str(result))
    
    # Record the exchange in the agent's manual chat history
    if should_use_history:
        agent.chat_history.add_user_message(task)
        agent.chat_history.add_ai_message(final_output)
    
    return {
        'required_skills': required_skills,
        'loaded_skills': loaded_skills,
        'response': final_output
    }


# ============================================================================
# SYSTEM PROMPT LOADING UTILITIES
# ============================================================================
# Helper functions that load Jinja2 prompt templates from disk and render
# them with skill context strings. Used during agent initialization or
# when dynamically swapping system prompts between calls.
# ============================================================================

def load_system_prompt(
    prompts_folder: Path,
    template_name: str = "system_prompt_template.jinja",
    skill_context: str = ""
) -> str:
    """
    Load a Jinja2 template from disk and render it with the given skill context.

    Args:
        prompts_folder: Directory containing Jinja2 template files.
        template_name: Filename of the template to load.
        skill_context: A pre-formatted string listing available skills,
                       injected into the template's ``SKILL_CONTEXT`` variable.

    Returns:
        The fully rendered system prompt string.
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
    Convenience function: load a system prompt template and automatically
    populate it with all available skills from the given skills directory.

    This reads every skill's metadata (name, description, path) and formats
    them into a Markdown-style list that is injected into the template's
    ``SKILL_CONTEXT`` variable.

    Args:
        prompts_folder: Directory containing Jinja2 template files.
        skills_folder: Root directory containing skill subdirectories.
        template_name: Filename of the template to load.

    Returns:
        The rendered system prompt string with all skill descriptions included.
    """
    from LabAgentSkill import skills_utils
    
    # Load metadata for every skill in the hub
    all_skills = skills_utils.read_all_skills_metadata(skills_folder)
    
    # Format each skill as a Markdown bullet with name, description, and file path
    skill_context = "\n".join([
        f"- **{skill['name']}**: {skill['description']} "
        f"(Path: {skills_folder / skill['name'] / 'SKILL.md'})"
        for skill in all_skills
    ])
    
    # Render the template with the assembled skill context
    return load_system_prompt(
        prompts_folder,
        template_name,
        skill_context
    )
