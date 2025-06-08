# src/llm_model.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from src.config import GEMINI_MODEL_NAME
from src.utils import logger

class GeminiLLM:
    """
    Manages the initialization and retrieval of the Google Gemini LLM.
    """
    def __init__(self, api_key: str | None = None):
        """
        Initializes the GeminiLLM handler.

        Args:
            api_key (str | None): Google API key. If None, it will try to
                                  read from GOOGLE_API_KEY environment variable.
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found. Please set it in .env or provide as argument.")
                raise ValueError("GOOGLE_API_KEY is required to initialize GeminiLLM.")
        self._api_key = api_key
        self._llm: BaseChatModel | None = None
        logger.info(f"GeminiLLM initialized with model: {GEMINI_MODEL_NAME}")

    def get_llm(self) -> BaseChatModel:
        """
        Returns an initialized ChatGoogleGenerativeAI instance.

        Returns:
            BaseChatModel: The Langchain ChatGoogleGenerativeAI model.
        """
        if self._llm is None:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_NAME,
                    google_api_key=self._api_key,
                    temperature=0.7, # Adjust creativity (0.0-1.0)
                    convert_system_message_to_human=True # Recommended for Gemini
                )
                logger.info(f"Successfully loaded Google Gemini LLM: {GEMINI_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Failed to load Google Gemini LLM: {e}")
                raise
        return self._llm
