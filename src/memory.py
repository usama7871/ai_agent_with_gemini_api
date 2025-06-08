# src/memory.py

from langchain.memory import ConversationBufferMemory
from langchain_core.memory import BaseMemory
from src.utils import logger

def get_conversation_memory() -> BaseMemory:
    """
    Initializes and returns a ConversationBufferMemory instance.
    This memory keeps the full conversation history.

    Returns:
        BaseMemory: A Langchain memory object.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history", # This key must match the agent's expected memory key
        return_messages=True # Return messages as list of BaseMessage objects
    )
    logger.info("ConversationBufferMemory initialized.")
    return memory

# For future scalability:
# If you needed persistent memory across sessions or users, you would
# replace ConversationBufferMemory with something like:
# from langchain.memory import RedisChatMessageHistory
# from langchain.memory import ConversationBufferWindowMemory
#
# def get_persistent_memory(session_id: str) -> BaseMemory:
#     # Example for Redis (requires Redis server and redis-py installed)
#     # history = RedisChatMessageHistory(session_id=session_id, url="redis://localhost:6379/0")
#     # return ConversationBufferMemory(chat_memory=history, memory_key="chat_history", return_messages=True)
#     pass
