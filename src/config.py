# src/config.py

import os

# --- LLM Configuration ---
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"
# You might want to use "gemini-2.0-flash" for multimodal tasks, but gemini-pro is text-only.

# --- Agent Configuration ---
AGENT_SYSTEM_PROMPT: str = """
You are a highly capable AI assistant named Gemini Agent.
You are designed to be helpful, friendly, and comprehensive.
You have access to various tools to gather information and complete tasks.
Always try to use your tools to get factual information when asked specific questions
or when external data might be needed.
If a question is a simple knowledge recall, you can answer directly.
Maintain a consistent friendly tone.
"""

# --- Logging Configuration ---
LOG_FILE: str = "logs/agent.log"
LOG_LEVEL: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Streamlit UI Configuration ---
APP_TITLE: str = "Gemini AI Agent Chatbot"
APP_ICON: str = "✨🤖" # E.g., "🤖", "🚀", "💬"


# --- Enhanced Memory Configuration ---
MEMORY_TYPE = "buffer"  # Options: "buffer", "window", "summary"
MEMORY_WINDOW_SIZE = 6  # For ConversationBufferWindowMemory
MAX_TOKEN_LIMIT = 2000  # For ConversationSummaryBufferMemory

# --- Tool Configuration ---
TOOL_TIMEOUT = 10  # Seconds before tool times out

# --- Session Management ---
SESSION_EXPIRATION = 3600  # 1 hour session expiration (for persistent memory)

# --- New UI Settings ---
ENABLE_MEMORY_MANAGEMENT = True
