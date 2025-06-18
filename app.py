# app.py

import os
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# Local imports
from src.utils import setup_logging, logger
from src.config import APP_TITLE, APP_ICON, MEMORY_TYPE, ENABLE_MEMORY_MANAGEMENT
from src.llm_model import GeminiLLM
from src.tools import get_agent_tools
from src.memory import get_conversation_memory
from src.agent import AIAgent
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random

# --- Initial Setup ---
load_dotenv()
setup_logging()

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="🌌 Galactic AI Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Futuristic CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a0b2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
        background-attachment: fixed;
        color: #e0e6ed;
        font-family: 'Exo 2', sans-serif;
    }
    
    /* Animated Stars Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.6), transparent),
            radial-gradient(2px 2px at 160px 30px, #fff, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes sparkle {
        from { transform: translateY(0px); }
        to { transform: translateY(-100px); }
    }
    
    /* Main Title */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00d4ff, #ff00ff, #00ff88, #ffaa00);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        margin-bottom: 1rem;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #a0a8b0;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Sidebar Enhancement */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 11, 46, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(0, 212, 255, 0.2);
        backdrop-filter: blur(5px);
    }
    
    .sidebar-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        margin-bottom: 1rem;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 15px !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 255, 0.1)) !important;
        border-color: rgba(0, 212, 255, 0.4) !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(255, 170, 0, 0.1)) !important;
        border-color: rgba(0, 255, 136, 0.4) !important;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 25px !important;
        color: #e0e6ed !important;
        font-family: 'Exo 2', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff00ff, #cc0099) !important;
        box-shadow: 0 5px 20px rgba(255, 0, 255, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 15px !important;
    }
    
    /* Status Indicators */
    .status-online {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-offline {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Glowing Effects */
    .glow-text {
        text-shadow: 0 0 10px currentColor;
    }
    
    .pulse-border {
        animation: pulse-border 2s infinite;
    }
    
    @keyframes pulse-border {
        0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.5); }
        50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 30px rgba(0, 212, 255, 0.4); }
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(0, 212, 255, 0.3);
        border-top: 3px solid #00d4ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Message Timestamps */
    .message-timestamp {
        font-size: 0.8rem;
        color: #7a8288;
        font-style: italic;
        text-align: right;
        margin-top: 0.5rem;
    }
    
    /* Statistics Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 255, 0.1));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    
    .stat-number {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #a0a8b0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {display: none;}
    
    /* Holographic Effects */
    .hologram-text {
        background: linear-gradient(45deg, transparent 30%, rgba(0, 212, 255, 0.5) 50%, transparent 70%);
        background-size: 200% 100%;
        animation: hologram 3s linear infinite;
        -webkit-background-clip: text;
        background-clip: text;
    }
    
    @keyframes hologram {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Neural Network Animation */
    .neural-network {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.1;
    }
    
    .neural-node {
        position: absolute;
        width: 4px;
        height: 4px;
        background: #00d4ff;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.5); opacity: 1; }
    }
    
    /* Advanced Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(255, 0, 255, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    
    /* Voice Mode Indicator */
    .voice-indicator {
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #ff00ff, #00ff88);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        box-shadow: 0 4px 20px rgba(255, 0, 255, 0.4);
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .voice-indicator:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 30px rgba(255, 0, 255, 0.6);
    }
    
    .voice-active {
        animation: voice-pulse 1s infinite;
    }
    
    @keyframes voice-pulse {
        0%, 100% { box-shadow: 0 4px 20px rgba(255, 0, 255, 0.4); }
        50% { box-shadow: 0 4px 40px rgba(255, 0, 255, 0.8), 0 0 60px rgba(255, 0, 255, 0.4); }
    }
    
    /* Advanced Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(26, 11, 46, 0.9));
        color: #00d4ff;
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 1001;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: all 0.3s;
        border: 1px solid rgba(0, 212, 255, 0.3);
        font-size: 0.9rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Performance Graph Container */
    .performance-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }
    
    /* Command Palette */
    .command-palette {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(15, 15, 35, 0.95);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(0, 212, 255, 0.5);
        border-radius: 20px;
        width: 500px;
        max-height: 400px;
        z-index: 1002;
        display: none;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7);
    }
    
    .command-palette.active {
        display: block;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translate(-50%, -60%); }
        to { opacity: 1; transform: translate(-50%, -50%); }
    }
    
    /* Personality Selector */
    .personality-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        color: #00d4ff;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .personality-chip:hover,
    .personality-chip.active {
        background: rgba(0, 212, 255, 0.2);
        border-color: #00d4ff;
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all required session state variables with enhanced tracking"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    
    if "memory_type" not in st.session_state:
        st.session_state.memory_type = MEMORY_TYPE
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = time.time()
    
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "galactic"
    
    if "auto_scroll" not in st.session_state:
        st.session_state.auto_scroll = True
    
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = {
            "response_times": [],
            "tool_usage": {},
            "error_count": 0,
            "successful_responses": 0
        }
    
    if "voice_mode" not in st.session_state:
        st.session_state.voice_mode = False
    
    if "dark_mode_intensity" not in st.session_state:
        st.session_state.dark_mode_intensity = 100
    
    if "agent_personality" not in st.session_state:
        st.session_state.agent_personality = "Professional"
    
    if "notification_sound" not in st.session_state:
        st.session_state.notification_sound = True

initialize_session_state()

# --- Enhanced UI Components ---
def render_header():
    """Render the futuristic header with animations"""
    st.markdown("""
    <div class="main-title">🌌 GALACTIC AI AGENT 🚀</div>
    <div class="subtitle">✨ Your Advanced AI Companion from the Stars ✨</div>
    """, unsafe_allow_html=True)

def render_status_indicator():
    """Render agent status with visual indicators"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.session_state.agent_initialized:
            st.markdown("""
            <div class="status-online">
                🟢 AGENT ONLINE - READY FOR MISSION
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-offline">
                🔴 AGENT OFFLINE - AWAITING INITIALIZATION
            </div>
            """, unsafe_allow_html=True)

def render_neural_network_background():
    """Render animated neural network background"""
    nodes_html = ""
    for i in range(20):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        delay = random.uniform(0, 2)
        nodes_html += f'''
        <div class="neural-node" style="left: {left}%; top: {top}%; animation-delay: {delay}s;"></div>
        '''
    
    st.markdown(f'''
    <div class="neural-network">
        {nodes_html}
    </div>
    ''', unsafe_allow_html=True)

def render_voice_mode_indicator():
    """Render voice mode toggle indicator"""
    if st.session_state.voice_mode:
        st.markdown('''
        <div class="voice-indicator voice-active" title="Voice Mode Active">
            🎤
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="voice-indicator" title="Click to enable Voice Mode">
            🔇
        </div>
        ''', unsafe_allow_html=True)

def render_performance_metrics():
    """Render advanced performance metrics with visualizations"""
    if not st.session_state.performance_metrics["response_times"]:
        return
    
    st.markdown("### 📊 **Neural Network Performance Analytics**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time chart
        response_times = st.session_state.performance_metrics["response_times"]
        fig = go.Figure(data=go.Scatter(
            y=response_times,
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#00d4ff', width=3),
            marker=dict(color='#ff00ff', size=8)
        ))
        
        fig.update_layout(
            title="Response Time Analysis",
            xaxis_title="Request Number",
            yaxis_title="Time (seconds)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed'),
            title_font=dict(color='#00d4ff')
        )
        
        fig.update_xaxes(gridcolor='rgba(0, 212, 255, 0.2)')
        fig.update_yaxes(gridcolor='rgba(0, 212, 255, 0.2)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tool usage pie chart
        tool_usage = st.session_state.performance_metrics["tool_usage"]
        if tool_usage:
            fig = px.pie(
                values=list(tool_usage.values()),
                names=list(tool_usage.keys()),
                title="Tool Usage Distribution",
                color_discrete_sequence=['#00d4ff', '#ff00ff', '#00ff88', '#ffaa00']
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                title_font=dict(color='#00d4ff')
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_advanced_stats_dashboard():
    """Render enhanced statistics dashboard with advanced metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    session_duration = int(time.time() - st.session_state.session_start_time)
    metrics = st.session_state.performance_metrics
    
    avg_response_time = (sum(metrics["response_times"]) / len(metrics["response_times"])) if metrics["response_times"] else 0
    success_rate = (metrics["successful_responses"] / (metrics["successful_responses"] + metrics["error_count"]) * 100) if (metrics["successful_responses"] + metrics["error_count"]) > 0 else 100
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-number">{st.session_state.message_count}</div>
            <div class="stat-label">🚀 TRANSMISSIONS</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-number">{session_duration//60}m</div>
            <div class="stat-label">⏱️ MISSION TIME</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-number">{len(st.session_state.chat_history)//2}</div>
            <div class="stat-label">💬 EXCHANGES</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-number">{avg_response_time:.1f}s</div>
            <div class="stat-label">⚡ AVG RESPONSE</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col5:
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-number">{success_rate:.0f}%</div>
            <div class="stat-label">✅ SUCCESS RATE</div>
        </div>
        ''', unsafe_allow_html=True)

def render_enhanced_sidebar():
    """Render the enhanced futuristic sidebar"""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🛸 CONTROL PANEL</div>', unsafe_allow_html=True)
        
        # API Key Configuration
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**🔑 NEURAL LINK CONFIGURATION**")
            
            api_key = st.text_input(
                "Gemini API Access Key",
                value=st.session_state.api_key,
                type="password",
                help="🔐 Secure connection to Gemini AI Core",
                placeholder="Enter your galactic access code..."
            )
            
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                st.session_state.agent_initialized = False
                st.rerun()
            
            # Connection Status
            if st.session_state.api_key:
                st.success("🟢 Neural Link Established")
            else:
                st.error("🔴 Neural Link Required")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Memory Configuration
        if ENABLE_MEMORY_MANAGEMENT:
            with st.container():
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("**🧠 MEMORY MATRIX**")
                
                mem_type = st.selectbox(
                    "Memory Configuration",
                    ["buffer", "window", "summary"],
                    index=["buffer", "window", "summary"].index(st.session_state.memory_type),
                    help="🔮 Choose your memory processing mode"
                )
                
                if mem_type != st.session_state.memory_type:
                    st.session_state.memory_type = mem_type
                    st.session_state.agent_initialized = False
                    st.rerun()
                
                # Memory Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ PURGE", help="Clear conversation history"):
                        st.session_state.chat_history = []
                        st.session_state.message_count = 0
                        if "agent_instance" in st.session_state:
                            st.session_state.agent_instance._memory.clear()
                        st.success("🟢 Memory Purged!")
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button("🔄 RESET", help="Initialize new session"):
                        st.session_state.session_id = str(uuid.uuid4())[:8]
                        st.session_state.agent_initialized = False
                        st.session_state.chat_history = []
                        st.session_state.message_count = 0
                        st.session_state.session_start_time = time.time()
                        st.success("🟢 System Reset!")
                        time.sleep(1)
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced Settings
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**⚙️ ADVANCED SETTINGS**")
            
            st.session_state.auto_scroll = st.checkbox(
                "🔄 Auto-Scroll Messages",
                value=st.session_state.auto_scroll,
                help="Automatically scroll to latest messages"
            )
            
            st.session_state.voice_mode = st.checkbox(
                "🎤 Voice Mode",
                value=st.session_state.voice_mode,
                help="Enable voice input/output capabilities"
            )
            
            st.session_state.notification_sound = st.checkbox(
                "🔊 Notification Sounds",
                value=st.session_state.notification_sound,
                help="Play sounds for notifications"
            )
            
            # Personality Selector
            st.markdown("**🤖 Agent Personality**")
            personalities = ["Professional", "Friendly", "Scientific", "Casual", "Enthusiastic"]
            
            personality_html = ""
            for personality in personalities:
                active_class = "active" if personality == st.session_state.agent_personality else ""
                personality_html += f'''
                <span class="personality-chip {active_class}" onclick="selectPersonality('{personality}')">
                    {personality}
                </span>
                '''
            
            st.markdown(f'<div>{personality_html}</div>', unsafe_allow_html=True)
            
            # Dark mode intensity
            st.session_state.dark_mode_intensity = st.slider(
                "🌌 Cosmic Intensity",
                min_value=50,
                max_value=100,
                value=st.session_state.dark_mode_intensity,
                help="Adjust the darkness of the galactic theme"
            )
            
            # Theme selector
            theme = st.selectbox(
                "🎨 Interface Theme",
                ["Galactic", "Cyberpunk", "Deep Space", "Neon City"],
                index=0,
                help="🌌 Choose your visual experience"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Session Information
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**📊 SESSION ANALYTICS**")
            
            st.markdown(f"""
            - **Session ID:** `{st.session_state.session_id}`
            - **Memory Type:** {st.session_state.memory_type.upper()}
            - **Status:** {'🟢 ACTIVE' if st.session_state.agent_initialized else '🔴 STANDBY'}
            - **Uptime:** {int(time.time() - st.session_state.session_start_time)//60}m {int(time.time() - st.session_state.session_start_time)%60}s
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_chat_history():
    """Display the conversation history with enhanced formatting"""
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #7a8288;">
            <h3>🌟 Welcome to the Galactic Communication Hub 🌟</h3>
            <p>Initialize your neural link and begin your journey through the cosmos of knowledge.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for i, message in enumerate(st.session_state.chat_history):
        timestamp = time.strftime("%H:%M", time.localtime())
        
        if message["type"] == "human":
            with st.chat_message("user", avatar="👨‍🚀"):
                st.markdown(message["content"])
                if st.session_state.auto_scroll:
                    st.markdown(f'<div class="message-timestamp">Transmitted at {timestamp}</div>', 
                              unsafe_allow_html=True)
        
        elif message["type"] == "ai":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                if st.session_state.auto_scroll:
                    st.markdown(f'<div class="message-timestamp">Received at {timestamp}</div>', 
                              unsafe_allow_html=True)

def initialize_agent() -> bool:
    """Initialize the AI agent with enhanced error handling and feedback"""
    try:
        if not st.session_state.api_key:
            st.warning("🔑 Neural Link Configuration Required")
            return False
        
        with st.spinner("🚀 Initializing Galactic AI Systems..."):
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize memory
            status_text.text("🧠 Configuring Memory Matrix...")
            progress_bar.progress(25)
            memory = get_conversation_memory(
                memory_type=st.session_state.memory_type,
                session_id=st.session_state.session_id
            )
            
            # Initialize LLM
            status_text.text("🤖 Establishing Neural Network Connection...")
            progress_bar.progress(50)
            llm = GeminiLLM(api_key=st.session_state.api_key).get_llm()
            
            # Initialize tools
            status_text.text("🛠️ Loading Galactic Tools...")
            progress_bar.progress(75)
            tools = get_agent_tools()
            
            # For summary memory, set the LLM
            if st.session_state.memory_type == "summary" and hasattr(memory, "llm"):
                memory.llm = llm
            
            # Create agent
            status_text.text("⚡ Finalizing Agent Initialization...")
            progress_bar.progress(90)
            st.session_state.agent_instance = AIAgent(
                llm=llm,
                tools=tools,
                memory=memory
            ).get_runnable_agent()
            
            progress_bar.progress(100)
            status_text.text("✅ Agent Successfully Initialized!")
            
            st.session_state.agent_initialized = True
            logger.info(f"Agent initialized for session {st.session_state.session_id}")
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎯 Galactic AI Agent is now online and ready for mission!")
            return True
    
    except Exception as e:
        st.error(f"❌ Initialization Failed: {str(e)}")
        logger.error(f"Agent initialization error: {str(e)}")
        return False

def handle_user_input(prompt: str):
    """Handle user input with enhanced processing and feedback"""
    start_time = time.time()
    
    # Add user message to history
    st.session_state.chat_history.append({
        "type": "human",
        "content": prompt,
        "timestamp": time.time()
    })
    st.session_state.message_count += 1
    
    # Display user message
    with st.chat_message("user", avatar="👨‍🚀"):
        st.markdown(prompt)
    
    # Get and display AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Processing through neural networks..."):
            try:
                # Show enhanced typing indicator
                typing_placeholder = st.empty()
                typing_placeholder.markdown("🔄 *Connecting to galactic database...*")
                
                response = st.session_state.agent_instance.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                typing_placeholder.empty()
                
                ai_response = response.get("output", "❌ Neural networks encountered an anomaly.")
                
                # Apply personality modifications
                ai_response = apply_personality_filter(ai_response, st.session_state.agent_personality)
                
                st.markdown(ai_response)
                
                # Calculate response time
                response_time = time.time() - start_time
                st.session_state.performance_metrics["response_times"].append(response_time)
                st.session_state.performance_metrics["successful_responses"] += 1
                
                # Track tool usage (simplified)
                if "search" in prompt.lower():
                    st.session_state.performance_metrics["tool_usage"]["WebSearch"] = st.session_state.performance_metrics["tool_usage"].get("WebSearch", 0) + 1
                elif "time" in prompt.lower():
                    st.session_state.performance_metrics["tool_usage"]["TimeQuery"] = st.session_state.performance_metrics["tool_usage"].get("TimeQuery", 0) + 1
                elif "calculate" in prompt.lower():
                    st.session_state.performance_metrics["tool_usage"]["Calculator"] = st.session_state.performance_metrics["tool_usage"].get("Calculator", 0) + 1
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    "type": "ai",
                    "content": ai_response,
                    "timestamp": time.time(),
                    "response_time": response_time
                })
                
                # Play notification sound if enabled
                if st.session_state.notification_sound:
                    st.markdown('''
                    <script>
                    const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmEfhfr...');
                    audio.play().catch(e => console.log('Audio play failed:', e));
                    </script>
                    ''', unsafe_allow_html=True)
                
                logger.info(f"Response generated for: {prompt[:50]}...")
            
            except Exception as e:
                error_msg = f"🚨 **System Alert**: Neural network disruption detected.\n\n*Error Details*: {str(e)}"
                st.error("❌ Communication Error")
                st.markdown(error_msg)
                
                # Track error
                st.session_state.performance_metrics["error_count"] += 1
                
                st.session_state.chat_history.append({
                    "type": "ai",
                    "content": error_msg,
                    "timestamp": time.time()
                })
                logger.error(f"Agent error: {str(e)}")

def apply_personality_filter(response: str, personality: str) -> str:
    """Apply personality modifications to AI responses"""
    personality_modifiers = {
        "Professional": lambda x: x,
        "Friendly": lambda x: f"😊 {x}",
        "Scientific": lambda x: f"🔬 Based on my analysis: {x}",
        "Casual": lambda x: x.replace("I", "I'd say I").replace(".", " 😄"),
        "Enthusiastic": lambda x: f"🎉 {x}! This is fascinating! ✨"
    }
    
    return personality_modifiers.get(personality, lambda x: x)(response)

def render_quick_actions():
    """Render quick action buttons for common commands"""
    st.markdown("### ⚡ **Quick Actions**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🌍 Current Events", help="Search for latest news"):
            st.session_state.quick_prompt = "What are the latest important news and current events?"
    
    with col2:
        if st.button("🧮 Calculator", help="Open calculator mode"):
            st.session_state.quick_prompt = "I need help with some calculations"
    
    with col3:
        if st.button("🕒 Time & Date", help="Get current time"):
            st.session_state.quick_prompt = "What's the current time and date?"
    
    with col4:
        if st.button("💡 Random Fact", help="Get an interesting fact"):
            st.session_state.quick_prompt = "Tell me an interesting random fact"
    
    # Handle quick prompts
    prompt= st.session_state.get('quick_prompt', None)
    quick_prompt = st.session_state.get('quick_prompt', None)
    if quick_prompt:
        handle_user_input(quick_prompt)
        del st.session_state.quick_prompt
    if hasattr(st.session_state, 'quick_prompt') and st.session_state.quick_prompt:
        handle_user_input(st.session_state.quick_prompt)
        del st.session_state.quick_prompt
    st.session_state.chat_history.append({
        "type": "human",
        "content": prompt,
        "timestamp": time.time()
    })
    st.session_state.message_count += 1
    
    # Display user message
    with st.chat_message("user", avatar="👨‍🚀"):
        st.markdown(prompt)
    
    # Get and display AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Processing through neural networks..."):
            try:
                # Show typing indicator
                typing_placeholder = st.empty()
                typing_placeholder.markdown("🔄 *Agent is thinking...*")
                
                response = st.session_state.agent_instance.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                typing_placeholder.empty()
                
                ai_response = response.get("output", "❌ Neural networks encountered an anomaly.")
                st.markdown(ai_response)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    "type": "ai",
                    "content": ai_response,
                    "timestamp": time.time()
                })
                
                logger.info(f"Response generated for: {prompt[:50]}...")
            
            except Exception as e:
                error_msg = f"🚨 **System Alert**: Neural network disruption detected.\n\n*Error Details*: {str(e)}"
                st.error("❌ Communication Error")
                st.markdown(error_msg)
                
                st.session_state.chat_history.append({
                    "type": "ai",
                    "content": error_msg,
                    "timestamp": time.time()
                })
                logger.error(f"Agent error: {str(e)}")

# --- Main Application ---
def main():
    # Inject custom CSS
    inject_custom_css()
    
    # Render header
    render_header()
    
    # Render status indicator
    render_status_indicator()
    
    # Main layout
    main_col, sidebar_col = st.columns([3, 1])
    
    with sidebar_col:
        render_enhanced_sidebar()
    
    with main_col:
        # Stats dashboard
        render_advanced_stats_dashboard()
        st.divider()
        
        # Chat interface
        render_chat_history()
        
        # Initialize agent if not done
        if not st.session_state.agent_initialized:
            if st.session_state.api_key:
                if st.button("🚀 INITIALIZE GALACTIC AGENT", 
                           help="Launch your AI companion", 
                           use_container_width=True):
                    initialize_agent()
        
        # Chat input
        if st.session_state.agent_initialized:
            if prompt := st.chat_input("🌟 Transmit your message to the galactic network..."):
                handle_user_input(prompt)
                
                # Auto-scroll to bottom if enabled
                if st.session_state.auto_scroll:
                    st.rerun()
        
        elif not st.session_state.api_key:
            st.info("🔑 **Neural Link Required**: Please configure your Gemini API key in the control panel to establish communication with the galactic network.")
        
        else:
            st.info("🚀 **Ready for Launch**: Click the initialization button above to activate your Galactic AI Agent.")

if __name__ == "__main__":
    main()
