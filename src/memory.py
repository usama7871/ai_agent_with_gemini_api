
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.memory import BaseMemory
from langchain_core.prompts import PromptTemplate
from src.config import MEMORY_WINDOW_SIZE, MAX_TOKEN_LIMIT
from src.utils import logger

SUMMARIZATION_PROMPT = PromptTemplate.from_template(
    "Progressively summarize the lines of conversation provided, "
    "adding onto the previous summary returning a new summary.\n\n"
    "Current summary:\n{summary}\n\n"
    "New lines of conversation:\n{new_lines}\n\n"
    "New summary:"
)

def get_conversation_memory(memory_type: str = "buffer", session_id: str = "default") -> BaseMemory:
    """
    Initializes and returns a memory instance with enhanced options.
    
    Args:
        memory_type (str): Type of memory ('buffer', 'window', 'summary')
        session_id (str): Session identifier for persistent memories
        
    Returns:
        BaseMemory: A Langchain memory object
    """
    if memory_type == "window":
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=MEMORY_WINDOW_SIZE,
            return_messages=True
        )
        logger.info(f"ConversationBufferWindowMemory initialized (window={MEMORY_WINDOW_SIZE}).")
    elif memory_type == "summary":
        memory = ConversationSummaryBufferMemory(
            memory_key="chat_history",
            max_token_limit=MAX_TOKEN_LIMIT,
            return_messages=True,
            llm=None,  # Will be set later when LLM is available
            prompt=SUMMARIZATION_PROMPT
        )
        logger.info(f"ConversationSummaryBufferMemory initialized (max_tokens={MAX_TOKEN_LIMIT}).")
    else:  # Default to buffer
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        logger.info("ConversationBufferMemory initialized.")
    
    return memory

# For Redis-based persistent memory (uncomment when needed)
def get_persistent_memory(session_id: str) -> BaseMemory:
    from langchain.memory import RedisChatMessageHistory
    history = RedisChatMessageHistory(
        session_id=session_id, 
        url="redis://localhost:6379/0"
    )
    return ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True
    )
