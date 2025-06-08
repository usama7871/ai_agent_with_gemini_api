# app.py

import streamlit as st
import os
from dotenv import load_dotenv

from src.utils import setup_logging, logger
from src.config import APP_TITLE, APP_ICON
from src.llm_model import GeminiLLM
from src.tools import get_agent_tools
from src.memory import get_conversation_memory
from src.agent import AIAgent
from langchain_core.messages import HumanMessage, AIMessage

# --- Load Environment Variables ---
load_dotenv()
logger.info(".env file loaded.")

# --- Setup Logging ---
setup_logging()

# --- Streamlit App Configuration ---
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="centered")
st.title(f"{APP_ICON} {APP_TITLE}")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Stores (role, content) tuples
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = None


# --- Sidebar for API Key Input (Optional but good for demos/testing) ---
with st.sidebar:
    st.header("Configuration")
    gemini_api_key = os.getenv("GOOGLE_API_KEY") # Try to get from .env first

    # If key is not in .env, allow user to input
    if not gemini_api_key:
        gemini_api_key = st.text_input(
            "Enter your Google Gemini API Key:",
            type="password",
            key="gemini_api_key_input",
            help="Get your key from Google AI Studio: https://makersuite.google.com/"
        )

    if gemini_api_key:
        # Validate and configure if key is provided
        try:
            # Initialize core components
            llm_handler = GeminiLLM(api_key=gemini_api_key)
            llm = llm_handler.get_llm()
            tools = get_agent_tools()
            memory = get_conversation_memory() # Memory is session-specific

            # Initialize and store the agent instance in session state
            st.session_state.agent_instance = AIAgent(llm=llm, tools=tools, memory=memory).get_runnable_agent()
            st.session_state.api_key_configured = True
            st.success("API Key configured and agent initialized!")
            logger.info("Agent successfully configured via Streamlit sidebar/env.")
        except ValueError as e:
            st.error(f"Error initializing agent: {e}. Please check your API key.")
            st.session_state.api_key_configured = False
            st.session_state.agent_instance = None
            logger.error(f"Agent initialization failed: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during agent setup: {e}")
            st.session_state.api_key_configured = False
            st.session_state.agent_instance = None
            logger.error(f"Unexpected error during agent setup: {e}")
    else:
        st.warning("Please enter your Google Gemini API Key to start chatting.")
        st.session_state.api_key_configured = False
        st.session_state.agent_instance = None

# --- Display Chat History ---
for message_type, message_content in st.session_state.chat_history:
    if message_type == "human":
        with st.chat_message("user"):
            st.markdown(message_content)
    elif message_type == "ai":
        with st.chat_message("assistant"):
            st.markdown(message_content)

# --- Chat Input ---
if st.session_state.api_key_configured:
    user_query = st.chat_input("Ask Gemini Agent anything...")
    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append(("human", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the agent
                    response = st.session_state.agent_instance.invoke({"input": user_query})
                    ai_response = response.get("output", "I could not generate a response.")
                    st.markdown(ai_response)
                    st.session_state.chat_history.append(("ai", ai_response))
                    logger.info(f"Agent responded to '{user_query[:50]}...': {ai_response[:50]}...")
                except Exception as e:
                    error_message = f"An error occurred while processing your request: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append(("ai", error_message))
                    logger.error(f"Error during agent invocation for '{user_query}': {e}", exc_info=True)
else:
    st.info("Please configure your Google Gemini API Key in the sidebar to begin chatting.")