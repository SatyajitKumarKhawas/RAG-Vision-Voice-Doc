import os
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components
import base64
import logging
import tempfile
import platform
import subprocess
import json
import re
from datetime import datetime

# Document processing imports (RAG)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# RAG chain imports
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Vision and Voice imports
from groq import Groq
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gemini-2.0-flash"

# Updated System prompts for mental health counseling
COUNSELOR_SYSTEM_PROMPT = """
You are a supportive, empathetic AI mental health counselor for university students. Your role is to provide emotional support, guidance, and resources in a non-judgmental, caring manner.

*CRITICAL COUNSELING GUIDELINES:*

1. *TONE & APPROACH*:
   - Be warm, empathetic, and non-judgmental
   - Use supportive language like "It sounds like...", "That must feel...", "Many students experience..."
   - Avoid diagnostic language or medical terms
   - Focus on validation and emotional support first

2. *RESPONSE STYLE*:
   - Start with emotional validation
   - Normalize their feelings ("This is completely understandable...")
   - Offer hope and reassurance
   - Suggest coping strategies gently
   - End with encouragement and support availability

3. *SAFETY PROTOCOLS*:
   - If you detect crisis language (suicide, self-harm), immediately suggest professional help
   - Always recommend counseling when appropriate
   - Provide campus counselor recommendations when needed

4. *NEVER*:
   - Diagnose mental health conditions
   - Prescribe medications
   - Use clinical/medical terminology
   - Rush to solutions without validation

Context from mental health resources: {context}
Student's concern: {question}

PROVIDE A SUPPORTIVE, EMPATHETIC COUNSELING RESPONSE:
"""

VISION_COUNSELOR_PROMPT = """You are a caring mental health counselor analyzing visual content from a student. 

Look at this image with empathy and understanding. This might be:
- Journal entries or notes expressing feelings
- Artwork or drawings reflecting emotions
- Photos that might indicate mood or wellbeing
- Academic stress indicators

Respond as a supportive counselor would:
- Acknowledge what you observe with empathy
- Validate any emotions you perceive
- Offer gentle insights about emotional patterns
- Suggest healthy coping strategies
- Be encouraging and non-judgmental

Start your response with something like "I can see that you're expressing..." or "From what you've shared visually..."
Keep your tone warm, supportive, and understanding. No clinical language - speak like a caring counselor would."""

# Mental Health Screening Questionnaires
PHQ9_QUESTIONS = [
    "Over the last 2 weeks, how often have you had little interest or pleasure in doing things?",
    "Over the last 2 weeks, how often have you felt down, depressed, or hopeless?",
    "Over the last 2 weeks, how often have you had trouble falling or staying asleep, or sleeping too much?",
    "Over the last 2 weeks, how often have you felt tired or had little energy?",
    "Over the last 2 weeks, how often have you had poor appetite or been overeating?",
    "Over the last 2 weeks, how often have you felt bad about yourself — or felt like a failure or let yourself or family down?",
    "Over the last 2 weeks, how often have you had trouble concentrating on things, such as reading or watching TV?",
    "Over the last 2 weeks, how often have you moved or spoken slowly, or been fidgety or restless?",
    "Over the last 2 weeks, how often have you had thoughts that you would be better off dead, or thoughts of hurting yourself?"
]

GAD7_QUESTIONS = [
    "Over the last 2 weeks, how often have you felt nervous, anxious, or on edge?",
    "Over the last 2 weeks, how often have you not been able to stop or control worrying?",
    "Over the last 2 weeks, how often have you worried too much about different things?",
    "Over the last 2 weeks, how often have you had trouble relaxing?",
    "Over the last 2 weeks, how often have you been so restless that it's hard to sit still?",
    "Over the last 2 weeks, how often have you become easily annoyed or irritable?",
    "Over the last 2 weeks, how often have you felt afraid as if something awful might happen?"
]

GHQ12_QUESTIONS = [
    "In the last few weeks, have you been able to concentrate on what you are doing?",
    "In the last few weeks, have you lost much sleep over worry?",
    "In the last few weeks, have you felt that you are playing a useful part in things?",
    "In the last few weeks, have you felt capable of making decisions about things?",
    "In the last few weeks, have you felt constantly under strain?",
    "In the last few weeks, have you felt you couldn't overcome your difficulties?",
    "In the last few weeks, have you been able to enjoy your normal day-to-day activities?",
    "In the last few weeks, have you been able to face up to problems?",
    "In the last few weeks, have you been feeling unhappy and depressed?",
    "In the last few weeks, have you been losing confidence in yourself?",
    "In the last few weeks, have you been thinking of yourself as a worthless person?",
    "In the last few weeks, have you been feeling reasonably happy, all things considered?"
]

RESPONSE_OPTIONS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
GHQ_RESPONSE_OPTIONS = ["Better than usual", "Same as usual", "Less than usual", "Much less than usual"]

# Counselor Database
COUNSELORS = {
    "anxiety_depression": {
        "name": "Dr. Meera Sharma",
        "phone": "+91-98765-43210",
        "qualification": "Ph.D. in Clinical Psychology, University of Delhi",
        "expertise": ["Anxiety disorders", "Depression management", "Cognitive Behavioral Therapy (CBT)"],
        "style": "Calm, empathetic, evidence-driven; prefers structured therapy sessions",
        "focus": "Academic stress, panic attacks, and depressive episodes"
    },
    "relationships_stress": {
        "name": "Mr. Arjun Malhotra",
        "phone": "+91-99123-45678",
        "qualification": "M.A. in Counselling Psychology, Tata Institute of Social Sciences (TISS)",
        "expertise": ["Relationship counselling", "Stress management", "Career confusion"],
        "style": "Friendly, approachable, non-judgmental; believes in open dialogue",
        "focus": "Peer pressure, relationship issues, and uncertainty about the future"
    },
    "severe_mental_health": {
        "name": "Dr. Farah Qureshi",
        "phone": "+91-98012-33445",
        "qualification": "MD in Psychiatry, AIIMS Delhi",
        "expertise": ["Severe depression", "Bipolar disorder", "Medication-based interventions"],
        "style": "Direct but compassionate; explains conditions in simple language",
        "focus": "Suicidal thoughts, mood disorders, or psychiatric emergencies"
    },
    "trauma_ptsd": {
        "name": "Ms. Radhika Sen",
        "phone": "+91-97999-88776",
        "qualification": "M.Sc. in Clinical Psychology, Christ University",
        "expertise": ["Trauma counselling", "PTSD", "Mindfulness and relaxation techniques"],
        "style": "Warm, nurturing, patient listener; uses art therapy and guided relaxation",
        "focus": "Bullying, harassment, family trauma, or social withdrawal"
    },
    "substance_adjustment": {
        "name": "Mr. Kabir Singh",
        "phone": "+91-91234-77890",
        "qualification": "M.A. in Rehabilitation Counselling, Jamia Millia Islamia",
        "expertise": ["Substance abuse recovery", "Disability adjustment", "Peer-support training"],
        "style": "Energetic, motivating, community-focused; encourages resilience through group support",
        "focus": "Addiction, adjustment issues, or long-term rehabilitation support"
    }
}

# Relaxation audio content generator
RELAXATION_SCRIPTS = {
    "breathing": """
    Let's do a simple breathing exercise together. Find a comfortable position and close your eyes if you feel comfortable.
    
    Take a slow, deep breath in through your nose for 4 counts... 1, 2, 3, 4.
    Now hold that breath gently for 2 counts... 1, 2.
    Slowly exhale through your mouth for 6 counts... 1, 2, 3, 4, 5, 6.
    
    Let's repeat this two more times. Breathe in... 1, 2, 3, 4. Hold... 1, 2. Out... 1, 2, 3, 4, 5, 6.
    
    One more time. In... 1, 2, 3, 4. Hold... 1, 2. Out... 1, 2, 3, 4, 5, 6.
    
    Notice how your body feels now. You're doing great. Remember, you can do this breathing exercise anytime you feel overwhelmed.
    """,
    
    "mindfulness": """
    Let's take a moment for mindful awareness. Sit comfortably and take three natural breaths.
    
    Now, notice five things you can see around you. Just observe them without judgment.
    Notice four things you can touch - the texture of your clothes, the temperature of the air.
    Listen for three sounds around you - maybe traffic, voices, or silence itself.
    Notice two things you can smell.
    And one thing you can taste.
    
    This is called grounding. It helps bring you back to the present moment when anxiety or worry takes over.
    You've just given yourself a gift of presence. Well done.
    """,
    
    "progressive_relaxation": """
    We're going to relax your body, one part at a time. Get comfortable and close your eyes if you'd like.
    
    Start by tensing your fists tightly for 5 seconds... now release and feel the relaxation.
    Tense your arms and shoulders... hold... now let go and feel the tension melting away.
    Scrunch up your face muscles... hold... now relax and feel your face soften.
    
    Tense your chest and back... hold... now release and feel your breathing deepen.
    Tighten your stomach muscles... hold... now let go completely.
    Tense your legs and feet... hold tight... now release and feel heavy and relaxed.
    
    Take a moment to notice how relaxed your body feels now. This is your natural state of calm.
    """
}

# Custom CSS for modern UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding: 0;
    }
    
    /* Navigation Styles */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .nav-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .nav-subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin: 0.5rem 0 1.5rem 0;
        font-size: 1rem;
    }
    
    .nav-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .nav-button {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .nav-button.active {
        background: white;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Chat Interface Styles */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        text-align: center;
    }
    
    .chat-messages {
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
    }
    
    .message {
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }
    
    .message.user {
        flex-direction: row-reverse;
    }
    
    .message-content {
        max-width: 70%;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    .message.user .message-content {
        background: #667eea;
        color: white;
        margin-right: 1rem;
    }
    
    .message.assistant .message-content {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        margin-left: 1rem;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    .message.user .message-avatar {
        background: #667eea;
        color: white;
    }
    
    .message.assistant .message-avatar {
        background: #e0e7ff;
        color: #667eea;
    }
    
    /* Voice Input Styles */
    .voice-container {
        background: white;
        border: 2px dashed #d1d5db;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .voice-container:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .voice-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 500;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    
    .voice-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .voice-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
    }
    
    .voice-transcript {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 60px;
        font-style: italic;
        color: #666;
    }
    
    /* Assessment Styles */
    .assessment-container {
        max-width: 900px;
        margin: 0 auto;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    .assessment-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        text-align: center;
    }
    
    .assessment-content {
        padding: 2rem;
    }
    
    .question-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .question-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-color: #667eea;
    }
    
    .question-text {
        font-size: 1.1rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .response-options {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    
    .response-option {
        flex: 1;
        min-width: 120px;
        padding: 0.8rem;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        cursor: pointer;
        text-align: center;
        transition: all 0.3s ease;
        background: white;
    }
    
    .response-option:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }
    
    .response-option.selected {
        border-color: #667eea;
        background: #667eea;
        color: white;
        font-weight: 500;
    }
    
    /* Counselor Cards */
    .counselor-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .counselor-card {
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .counselor-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .counselor-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        text-align: center;
    }
    
    .counselor-content {
        padding: 1.5rem;
    }
    
    .counselor-name {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .counselor-phone {
        font-size: 1.1rem;
        margin: 0.5rem 0;
        opacity: 0.9;
    }
    
    .counselor-detail {
        margin: 1rem 0;
    }
    
    .counselor-label {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    /* Utility Classes */
    .text-center { text-align: center; }
    .mb-3 { margin-bottom: 1rem; }
    .mt-3 { margin-top: 1rem; }
    .p-3 { padding: 1rem; }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .nav-buttons {
            flex-direction: column;
            align-items: center;
        }
        
        .nav-button {
            width: 200px;
        }
        
        .message-content {
            max-width: 85%;
        }
        
        .response-options {
            flex-direction: column;
        }
        
        .response-option {
            min-width: auto;
        }
    }
    
    /* Crisis Alert Styles */
    .crisis-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }
    
    /* Success Alert Styles */
    .success-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Info Alert Styles */
    .info-alert {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Voice input HTML/JS component
def create_modern_voice_input():
    """Create modern voice input component"""
    voice_html = """
    <div class="voice-container">
        <h3 style="margin: 0 0 1rem 0; color: #667eea;">🎤 Voice Input</h3>
        
        <button id="startBtn" onclick="startRecording()" class="voice-button">
            🎤 Start Recording
        </button>
        
        <button id="stopBtn" onclick="stopRecording()" disabled class="voice-button" style="background: #ff6b6b;">
            🛑 Stop Recording
        </button>
        
        <div id="status" style="margin: 1rem 0; font-weight: 500; color: #666;"></div>
        
        <div id="transcript" class="voice-transcript">
            Your transcribed text will appear here...
        </div>
        
        <button id="sendBtn" onclick="sendToChat()" disabled class="voice-button" style="background: #51cf66;">
            📤 Send to Chat
        </button>
        
        <button id="clearBtn" onclick="clearTranscript()" class="voice-button" style="background: #ffa502;">
            🗑️ Clear
        </button>
    </div>

    <script>
    let recognition = null;
    let isRecording = false;
    let finalTranscript = '';

    // Check if browser supports speech recognition
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
        recognition = new SpeechRecognition();
    }

    if (recognition) {
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            isRecording = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('status').innerHTML = '🔴 Recording... Speak now!';
            document.getElementById('status').style.color = '#ff6b6b';
        };

        recognition.onresult = function(event) {
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }
            
            document.getElementById('transcript').innerHTML = 
                finalTranscript + '<span style="color: #999;">' + interimTranscript + '</span>';
        };

        recognition.onerror = function(event) {
            document.getElementById('status').innerHTML = '❌ Error: ' + event.error;
            document.getElementById('status').style.color = '#ff6b6b';
            resetButtons();
        };

        recognition.onend = function() {
            isRecording = false;
            resetButtons();
            if (finalTranscript.trim() !== '') {
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('status').innerHTML = '✅ Recording completed!';
                document.getElementById('status').style.color = '#51cf66';
            } else {
                document.getElementById('status').innerHTML = '⚠️ No speech detected';
                document.getElementById('status').style.color = '#ffa502';
            }
        };
    } else {
        document.getElementById('status').innerHTML = '❌ Speech recognition not supported';
        document.getElementById('startBtn').disabled = true;
    }

    function startRecording() {
        if (recognition && !isRecording) {
            finalTranscript = '';
            document.getElementById('transcript').innerHTML = 'Listening...';
            document.getElementById('sendBtn').disabled = true;
            recognition.start();
        }
    }

    function stopRecording() {
        if (recognition && isRecording) {
            recognition.stop();
        }
    }

    function resetButtons() {
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }

    function sendToChat() {
        if (finalTranscript.trim() !== '') {
            parent.postMessage({
                type: 'voice_input',
                message: finalTranscript.trim()
            }, '*');
            
            document.getElementById('status').innerHTML = '📤 Sent to chat!';
            document.getElementById('status').style.color = '#51cf66';
        }
    }

    function clearTranscript() {
        finalTranscript = '';
        document.getElementById('transcript').innerHTML = 'Your transcribed text will appear here...';
        document.getElementById('sendBtn').disabled = true;
        document.getElementById('status').innerHTML = '';
    }
    </script>
    """
    return voice_html

# Navigation component
def render_navigation():
    """Render modern navigation"""
    st.markdown("""
    <div class="nav-container">
        <h1 class="nav-title">🧠 Student Mental Health AI Counselor</h1>
        <p class="nav-subtitle">A safe space for emotional support, guidance, and mental health resources</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("💬 RAG Chat Support", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    
    with col2:
        if st.button("🎨 Visual Chat Support", use_container_width=True):
            st.session_state.current_page = "visual"
            st.rerun()
    
    with col3:
        if st.button("📋 Mental Health Screening", use_container_width=True):
            st.session_state.current_page = "screening"
            st.rerun()
    
    with col4:
        if st.button("🧘 Guided Relaxation", use_container_width=True):
            st.session_state.current_page = "relaxation"
            st.rerun()
    
    with col5:
        if st.button("👥 Campus Counselors", use_container_width=True):
            st.session_state.current_page = "counselors"
            st.rerun()
    
    st.markdown("---")

# Utility Functions
@st.cache_data
def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_query(query, model, encoded_image, api_key):
    """Analyze image using GROQ API."""
    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    
    return chat_completion.choices[0].message.content

def transcribe_with_groq(stt_model, audio_filepath, api_key):
    """Transcribe audio using GROQ API."""
    client = Groq(api_key=api_key)
    
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    
    return transcription.text

def text_to_speech_with_gtts(input_text, output_filepath, slow_speed=True):
    """Convert text to speech using gTTS with slower, calmer pace."""
    language = "en"
    
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=slow_speed
    )
    audioobj.save(output_filepath)
    return output_filepath

def text_to_speech_with_elevenlabs(input_text, output_filepath, api_key):
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=api_key)

        response = client.text_to_speech.convert(
            voice_id="Rachel",
            model_id="eleven_turbo_v2",
            text=input_text
        )

        with open(output_filepath, "wb") as f:
            for chunk in response:
                f.write(chunk)

        return output_filepath
    except Exception as e:
        st.error(f"ElevenLabs TTS failed: {e}")
        return text_to_speech_with_gtts(input_text, output_filepath)

# Mental Health Screening Functions
def calculate_phq9_score(responses):
    """Calculate PHQ-9 depression severity score"""
    score = sum(responses)
    if score <= 4:
        return score, "Minimal depression"
    elif score <= 9:
        return score, "Mild depression"
    elif score <= 14:
        return score, "Moderate depression"
    elif score <= 19:
        return score, "Moderately severe depression"
    else:
        return score, "Severe depression"

def calculate_gad7_score(responses):
    """Calculate GAD-7 anxiety severity score"""
    score = sum(responses)
    if score <= 4:
        return score, "Minimal anxiety"
    elif score <= 9:
        return score, "Mild anxiety"
    elif score <= 14:
        return score, "Moderate anxiety"
    else:
        return score, "Severe anxiety"

def calculate_ghq12_score(responses):
    """Calculate GHQ-12 psychological distress score"""
    binary_responses = []
    for resp in responses:
        if resp in [0, 1]:
            binary_responses.append(0)
        else:
            binary_responses.append(1)
    
    score = sum(binary_responses)
    if score <= 2:
        return score, "Low psychological distress"
    elif score <= 5:
        return score, "Moderate psychological distress"
    else:
        return score, "High psychological distress"

def recommend_counselor(assessment_results, user_concerns=""):
    """Recommend appropriate counselor based on assessment and concerns"""
    concerns_lower = user_concerns.lower()
    
    # Check for crisis keywords first
    crisis_keywords = ["suicide", "kill", "die", "hurt myself", "end it all", "no point"]
    if any(keyword in concerns_lower for keyword in crisis_keywords):
        return "severe_mental_health"
    
    # Check for trauma/PTSD keywords
    trauma_keywords = ["trauma", "abuse", "assault", "bullying", "harassment", "ptsd"]
    if any(keyword in concerns_lower for keyword in trauma_keywords):
        return "trauma_ptsd"
    
    # Check for substance abuse keywords
    substance_keywords = ["alcohol", "drugs", "drinking", "smoking", "addiction"]
    if any(keyword in concerns_lower for keyword in substance_keywords):
        return "substance_adjustment"
    
    # Check for relationship keywords
    relationship_keywords = ["relationship", "breakup", "girlfriend", "boyfriend", "family", "friends", "lonely"]
    if any(keyword in concerns_lower for keyword in relationship_keywords):
        return "relationships_stress"
    
    # Based on assessment scores
    if 'phq9' in assessment_results or 'gad7' in assessment_results:
        phq9_score = assessment_results.get('phq9', {}).get('score', 0)
        gad7_score = assessment_results.get('gad7', {}).get('score', 0)
        
        if phq9_score >= 15 or gad7_score >= 15:
            return "severe_mental_health"
        elif phq9_score >= 5 or gad7_score >= 5:
            return "anxiety_depression"
    
    return "relationships_stress"

# Document Processing Classes
class DocumentProcessor:
    """Handles PDF loading and processing"""
    
    @staticmethod
    def load_pdf_files(data_path):
        """Load PDF files from directory"""
        if not os.path.exists(data_path):
            st.error(f"Data directory '{data_path}' not found!")
            return []
            
        loader = DirectoryLoader(
            data_path,
            glob='*.pdf',
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return documents
    
    @staticmethod
    def create_chunks(documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(documents)
        return text_chunks
    
    @staticmethod
    def create_vectorstore(text_chunks):
        """Create and save FAISS vectorstore"""
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        
        st.success(f"Vectorstore created successfully with {len(text_chunks)} chunks!")
        return db

class MentalHealthCounselor:
    """Mental Health RAG counselor class"""
    
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.setup_chain()

    @st.cache_resource
    def get_vectorstore(_self):
        """Load vectorstore with caching"""
        if not os.path.exists(DB_FAISS_PATH):
            return None
            
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        try:
            db = FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            return db
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            return None
    
    def setup_chain(self):
        """Setup the QA chain"""
        self.vectorstore = self.get_vectorstore()
        
        if self.vectorstore is None:
            return
        
        prompt = PromptTemplate(
            template=COUNSELOR_SYSTEM_PROMPT, 
            input_variables=["context", "question"]
        )
        
        try:
            google_api_key = st.secrets.get("GOOGLE_API_KEY")

            if not google_api_key:
                st.error("Google API Key not found in Streamlit Secrets.")
                return

            llm = ChatGoogleGenerativeAI(
                model=DEFAULT_MODEL,
                temperature=0.3,
                google_api_key=google_api_key
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': prompt}
            )
            
        except Exception as e:
            st.error(f"Error setting up Gemini API: {str(e)}")
    
    def get_response(self, query):
        """Get empathetic response from QA chain"""
        if self.qa_chain is None:
            return "I'm here to listen and support you. It seems there might be a technical issue right now, but please know that your feelings are valid and there are people who want to help.", []
        
        try:
            response = self.qa_chain.invoke({'query': query})
            return response["result"], response["source_documents"]
        except Exception as e:
            return f"I'm experiencing some technical difficulties, but I want you to know that what you're feeling matters. Please consider reaching out to one of our campus counselors for immediate support.", []

class VisionProcessor:
    """Handles emotional analysis of visual content"""

    def __init__(self, groq_api_key, elevenlabs_api_key=None):
        self.groq_api_key = groq_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
    
    def analyze_emotional_content(self, image_path, user_context=""):
        """Analyze emotional content in uploaded images"""
        try:
            encoded_image = encode_image(image_path)
            full_query = VISION_COUNSELOR_PROMPT
            if user_context:
                full_query += f"\n\nStudent's context: {user_context}"
            
            response = analyze_image_with_query(
                query=full_query,
                encoded_image=encoded_image,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                api_key=self.groq_api_key
            )
            return response
        except Exception as e:
            return f"I'm having trouble analyzing the image right now, but I want you to know that I'm here to listen. Sometimes technical issues happen, but your feelings and experiences are always valid. Would you like to tell me about what you wanted to share instead?"
    
    def generate_calming_audio(self, text, use_elevenlabs=False):
        """Generate calming audio response"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                output_path = temp_audio.name
            
            if use_elevenlabs and self.elevenlabs_api_key:
                text_to_speech_with_elevenlabs(text, output_path, self.elevenlabs_api_key)
            else:
                text_to_speech_with_gtts(text, output_path, slow_speed=True)
            
            return output_path
        except Exception as e:
            st.error(f"Error generating audio: {e}")
            return None

def detect_crisis_language(text):
    """Detect crisis or emergency language in user input"""
    crisis_keywords = [
        "suicide", "kill myself", "end my life", "want to die", "hurt myself", 
        "no point living", "better off dead", "can't go on", "end it all",
        "hopeless", "worthless", "nobody cares", "give up"
    ]
    
    text_lower = text.lower()
    crisis_detected = any(keyword in text_lower for keyword in crisis_keywords)
    
    return crisis_detected

def generate_crisis_response():
    """Generate immediate crisis support response"""
    return """
    🚨 **I'm really concerned about you right now, and I want you to know that your life has value and meaning.**
    
    **Immediate Support:**
    - **Emergency:** Call 112 or go to your nearest emergency room
    - **Crisis Helpline:** Call 1860-2662-345 (iCALL)
    - **Campus Emergency:** Contact campus security immediately
    
    **Please reach out to Dr. Farah Qureshi immediately:**
    - Phone: +91-98012-33445
    - She specializes in psychiatric emergencies and is available for crisis intervention
    
    You don't have to face this alone. There are people who care about you and want to help. Please reach out to someone right now - a friend, family member, counselor, or crisis helpline.
    
    Your feelings are temporary, but your life is precious. 💙
    """

def generate_relaxation_audio(script_type, use_elevenlabs=False, elevenlabs_api_key=None):
    """Generate guided relaxation audio"""
    if script_type not in RELAXATION_SCRIPTS:
        return None
    
    script = RELAXATION_SCRIPTS[script_type]
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            output_path = temp_audio.name
        
        if use_elevenlabs and elevenlabs_api_key:
            text_to_speech_with_elevenlabs(script, output_path, elevenlabs_api_key)
        else:
            text_to_speech_with_gtts(script, output_path, slow_speed=True)
        
        return output_path
    except Exception as e:
        st.error(f"Error generating relaxation audio: {e}")
        return None

# Page Components
def render_chat_page():
    """Render the chat support page"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 style="margin: 0;">💬 RAG Chat Support</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Share your thoughts and feelings. I'm here to listen and support you.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Voice input section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Voice input component
        components.html(create_modern_voice_input(), height=400)
    
    with col2:
        st.markdown("### Quick Support")
        if st.button("😰 I'm feeling anxious", use_container_width=True):
            add_message("user", "I'm feeling really anxious right now and I don't know what to do.")
            st.rerun()
        
        if st.button("😔 I'm feeling down", use_container_width=True):
            add_message("user", "I've been feeling really down lately and nothing seems to help.")
            st.rerun()
        
        if st.button("😵 I'm overwhelmed", use_container_width=True):
            add_message("user", "I'm feeling completely overwhelmed with everything going on.")
            st.rerun()
        
        if st.button("😟 I need someone to talk to", use_container_width=True):
            add_message("user", "I just need someone to talk to right now. I'm not feeling great.")
            st.rerun()
    
    st.markdown("---")
    
    # Chat messages display
    render_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("Share what's on your mind... I'm here to listen 💙"):
        handle_chat_input(prompt)

def render_chat_messages():
    """Render chat messages in modern style"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="message user">
                <div class="message-avatar">U</div>
                <div class="message-content">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Handle different message types
            if message.get('type') == 'crisis_response':
                st.markdown(f'<div class="crisis-alert">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message assistant">
                    <div class="message-avatar">AI</div>
                    <div class="message-content">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display audio if available
            if message.get('audio_path') and os.path.exists(message['audio_path']):
                st.audio(message['audio_path'], format="audio/mp3")

def add_message(role, content, message_type=None, audio_path=None):
    """Add message to chat history"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    message = {
        'role': role,
        'content': content
    }
    
    if message_type:
        message['type'] = message_type
    if audio_path:
        message['audio_path'] = audio_path
    
    st.session_state.messages.append(message)

def handle_chat_input(prompt):
    """Handle chat input and generate response"""
    # Check for crisis language first
    if detect_crisis_language(prompt):
        add_message('user', prompt)
        crisis_response = generate_crisis_response()
        add_message('assistant', crisis_response, 'crisis_response')
        st.rerun()
    
    # Add user message
    add_message('user', prompt)
    
    # Process with Mental Health Counselor
    if 'counselor' in st.session_state:
        with st.spinner("Processing your message..."):
            result, source_docs = st.session_state.counselor.get_response(prompt)
            
            # Generate audio response if vision processor available
            audio_path = None
            if 'vision_processor' in st.session_state:
                audio_path = st.session_state.vision_processor.generate_calming_audio(
                    result, use_elevenlabs=True
                )
            
            add_message('assistant', result, audio_path=audio_path)
    else:
        add_message('assistant', "I'm here to support you, but the counseling system isn't fully loaded yet. Your feelings are valid, and I encourage you to reach out to our campus counselors directly.")
    
    st.rerun()

def render_screening_page():
    """Render the mental health screening page"""
    st.markdown("""
    <div class="assessment-container">
        <div class="assessment-header">
            <h2 style="margin: 0;">📋 Mental Health Screening</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Confidential assessments to help understand your mental health and guide you to appropriate support</p>
        </div>
        <div class="assessment-content">
    """, unsafe_allow_html=True)
    
    # Assessment selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0;">
            <h3 style="color: #667eea; margin: 0 0 1rem 0;">🫁 Breathing Exercise</h3>
            <p style="color: #666; margin: 0 0 1.5rem 0;">4-7-8 breathing technique for immediate calm</p>
        """, unsafe_allow_html=True)
        
        if st.button("🎧 Start Breathing Exercise", use_container_width=True):
            audio_path = generate_relaxation_audio("breathing", use_elevenlabs=True, elevenlabs_api_key=st.secrets.get("ELEVENLABS_API_KEY"))
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
                st.markdown('<div class="success-alert">💙 Take your time with this exercise. Breathe at your own pace.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0;">
            <h3 style="color: #667eea; margin: 0 0 1rem 0;">🧠 Mindfulness</h3>
            <p style="color: #666; margin: 0 0 1.5rem 0;">5-4-3-2-1 grounding technique</p>
        """, unsafe_allow_html=True)
        
        if st.button("🎧 Start Mindfulness Exercise", use_container_width=True):
            audio_path = generate_relaxation_audio("mindfulness", use_elevenlabs=True, elevenlabs_api_key=st.secrets.get("ELEVENLABS_API_KEY"))
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
                st.markdown('<div class="success-alert">💙 Notice the present moment without judgment.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0;">
            <h3 style="color: #667eea; margin: 0 0 1rem 0;">😌 Progressive Relaxation</h3>
            <p style="color: #666; margin: 0 0 1.5rem 0;">Full body tension release</p>
        """, unsafe_allow_html=True)
        
        if st.button("🎧 Start Relaxation Exercise", use_container_width=True):
            audio_path = generate_relaxation_audio("progressive_relaxation", use_elevenlabs=True, elevenlabs_api_key=st.secrets.get("ELEVENLABS_API_KEY"))
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
                st.markdown('<div class="success-alert">💙 Let your body find its natural state of relaxation.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-alert" style="margin: 2rem 0;">
        <h4 style="margin: 0 0 0.5rem 0;">💡 Tip</h4>
        <p style="margin: 0;">Regular practice of these techniques can help build resilience against stress and anxiety. Even 5 minutes a day can make a difference!</p>
    </div>
    """, unsafe_allow_html=True)

def render_counselors_page():
    """Render the campus counselors page"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 style="margin: 0;">👥 Campus Counselors</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Our professional counselors are here to support you through any challenges</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display counselors in a grid
    st.markdown('<div class="counselor-grid">', unsafe_allow_html=True)
    
    for key, counselor in COUNSELORS.items():
        st.markdown(f"""
        <div class="counselor-card">
            <div class="counselor-header">
                <h3 class="counselor-name">{counselor['name']}</h3>
                <div class="counselor-phone">{counselor['phone']}</div>
            </div>
            <div class="counselor-content">
                <div class="counselor-detail">
                    <div class="counselor-label">🎓 Qualification</div>
                    <div>{counselor['qualification']}</div>
                </div>
                
                <div class="counselor-detail">
                    <div class="counselor-label">🔍 Expertise</div>
                    <div>{', '.join(counselor['expertise'])}</div>
                </div>
                
                <div class="counselor-detail">
                    <div class="counselor-label">💬 Counseling Style</div>
                    <div>{counselor['style']}</div>
                </div>
                
                <div class="counselor-detail">
                    <div class="counselor-label">🎯 Best For</div>
                    <div>{counselor['focus']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Emergency contacts
    st.markdown("""
    <div class="crisis-alert" style="margin: 2rem 0;">
        <h4 style="margin: 0 0 1rem 0;">🚨 Emergency Contacts</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div>
                <strong>Emergency Services</strong><br>
                📞 112
            </div>
            <div>
                <strong>Crisis Helpline</strong><br>
                📞 1860-2662-345 (iCALL)
            </div>
            <div>
                <strong>Campus Security</strong><br>
                📞 Available 24/7
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize database check
def initialize_system():
    """Initialize the mental health system"""
    # Check if vectorstore exists
    vectorstore_exists = os.path.exists(DB_FAISS_PATH)
    
    # Get API keys
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")
    
    # Initialize components
    if google_api_key and 'counselor' not in st.session_state:
        if vectorstore_exists:
            st.session_state.counselor = MentalHealthCounselor()
        else:
            # Show database setup in sidebar
            with st.sidebar:
                st.warning("⚠️ Mental Health Database Not Found")
                st.write("Please load the mental health resources to enable full counseling functionality.")
                
                if st.button("📚 Process Mental Health PDFs"):
                    with st.spinner("Processing mental health resources..."):
                        documents = DocumentProcessor.load_pdf_files(DATA_PATH)
                        
                        if not documents:
                            st.error("No PDF files found! Please add Gale Encyclopedia of Mental Health PDFs to the data directory.")
                        else:
                            text_chunks = DocumentProcessor.create_chunks(documents)
                            DocumentProcessor.create_vectorstore(text_chunks)
                            st.rerun()
    
    if groq_api_key and 'vision_processor' not in st.session_state:
        st.session_state.vision_processor = VisionProcessor(groq_api_key, elevenlabs_api_key)
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'chat'
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

# Handle voice input from JavaScript
def handle_voice_input():
    """Handle voice input from the component"""
    # This would be handled by the JavaScript postMessage in a real implementation
    # For now, we'll use a simple session state approach
    pass

def main():
    st.set_page_config(
        page_title="Student Mental Health AI Counselor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize system
    initialize_system()
    
    # Render navigation
    render_navigation()
    
    # Route to appropriate page
    current_page = st.session_state.get('current_page', 'chat')
    
    if current_page == 'chat':
        render_chat_page()
    elif current_page == 'visual':
        render_visual_page()
    elif current_page == 'screening':
        render_screening_page()
    elif current_page == 'relaxation':
        render_relaxation_page()
    elif current_page == 'counselors':
        render_counselors_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;'>
        <div style='margin-bottom: 1rem;'>
            <strong style='color: #667eea;'>💙 You're not alone.</strong> This AI counselor provides support, but professional counselors are always available for deeper help.
        </div>
        <div style='margin-bottom: 1rem;'>
            <strong style='color: #667eea;'>🔒 Confidential:</strong> Your conversations here are private and not stored permanently.
        </div>
        <div>
            <strong style='color: #ff6b6b;'>🚨 Crisis?</strong> Call 112 (Emergency) or 1860-2662-345 (iCALL Crisis Helpline) immediately.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() st.columns(3)
    
    with col1:
        if st.button("📊 PHQ-9 Depression Screening", use_container_width=True, type="primary"):
            st.session_state.selected_assessment = "PHQ-9"
            st.rerun()
    
    with col2:
        if st.button("😰 GAD-7 Anxiety Screening", use_container_width=True, type="primary"):
            st.session_state.selected_assessment = "GAD-7"
            st.rerun()
    
    with col3:
        if st.button("🔍 GHQ-12 General Health", use_container_width=True, type="primary"):
            st.session_state.selected_assessment = "GHQ-12"
            st.rerun()
    
    # Display selected assessment
    if hasattr(st.session_state, 'selected_assessment'):
        create_modern_screening_interface(st.session_state.selected_assessment)
    
    # Display results if available
    display_modern_assessment_results()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def create_modern_screening_interface(test_type):
    """Create modern interface for mental health screening tests"""
    if test_type == "PHQ-9":
        questions = PHQ9_QUESTIONS
        options = RESPONSE_OPTIONS
        title = "📊 PHQ-9 Depression Screening"
        description = "This questionnaire helps assess depression symptoms over the past 2 weeks."
        
    elif test_type == "GAD-7":
        questions = GAD7_QUESTIONS
        options = RESPONSE_OPTIONS
        title = "😰 GAD-7 Anxiety Screening"
        description = "This questionnaire helps assess anxiety symptoms over the past 2 weeks."
        
    elif test_type == "GHQ-12":
        questions = GHQ12_QUESTIONS
        options = GHQ_RESPONSE_OPTIONS
        title = "🔍 GHQ-12 General Health Screening"
        description = "This questionnaire assesses your recent psychological wellbeing."
    
    st.markdown(f"""
    <div style="margin: 2rem 0;">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: #666; font-style: italic; margin-bottom: 1.5rem;">{description}</p>
        <p style="color: #888; font-size: 0.9rem; margin-bottom: 2rem;">Please answer honestly. Your responses are confidential and will help us understand how to support you better.</p>
    </div>
    """, unsafe_allow_html=True)
    
    responses = []
    
    with st.form(f"{test_type}_modern_form"):
        for i, question in enumerate(questions):
            st.markdown(f"""
            <div class="question-card">
                <div class="question-text">{question}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create custom radio buttons
            response = st.radio(
                f"Question {i+1}",
                options=list(range(len(options))),
                format_func=lambda x: options[x],
                key=f"{test_type}_modern_{i}",
                horizontal=True,
                label_visibility="collapsed"
            )
            responses.append(response)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        submitted = st.form_submit_button(
            "Submit Assessment", 
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            if test_type == "PHQ-9":
                score, severity = calculate_phq9_score(responses)
                st.session_state[f'{test_type.lower()}_results'] = {
                    'score': score,
                    'severity': severity,
                    'responses': responses
                }
            elif test_type == "GAD-7":
                score, severity = calculate_gad7_score(responses)
                st.session_state[f'{test_type.lower()}_results'] = {
                    'score': score,
                    'severity': severity,
                    'responses': responses
                }
            elif test_type == "GHQ-12":
                score, severity = calculate_ghq12_score(responses)
                st.session_state[f'{test_type.lower()}_results'] = {
                    'score': score,
                    'severity': severity,
                    'responses': responses
                }
            
            st.rerun()

def display_modern_assessment_results():
    """Display assessment results with modern styling"""
    results = {}
    
    # Collect all assessment results
    for test in ['phq-9', 'gad-7', 'ghq-12']:
        if f'{test}_results' in st.session_state:
            results[test.replace('-', '')] = st.session_state[f'{test}_results']
    
    if not results:
        return None
    
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h3 style="color: #667eea;">📊 Your Assessment Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display results with empathetic framing
    for test_name, result in results.items():
        score = result['score']
        severity = result['severity']
        
        # Color based on severity
        if "severe" in severity.lower():
            color = "#ff6b6b"
            bg_color = "#ffe0e0"
        elif "moderate" in severity.lower():
            color = "#ffa502"
            bg_color = "#fff4e0"
        else:
            color = "#51cf66"
            bg_color = "#e8f5e8"
        
        st.markdown(f"""
        <div style="
            background: {bg_color};
            border-left: 4px solid {color};
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
        ">
            <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{test_name.upper()} Results</h4>
            <p style="margin: 0 0 1rem 0; font-weight: 600;">Score: {score} - {severity}</p>
        """, unsafe_allow_html=True)
        
        # Provide empathetic interpretation
        if "severe" in severity.lower():
            st.markdown("""
            <p style="margin: 0;">💙 Your results suggest you might be experiencing significant distress. Please know that you're not alone, and seeking support is a sign of strength. I'd recommend speaking with one of our professional counselors.</p>
            """, unsafe_allow_html=True)
        elif "moderate" in severity.lower():
            st.markdown("""
            <p style="margin: 0;">💙 Your results indicate you might be experiencing some challenges. It's completely normal to go through difficult times, and there are effective ways to help you feel better.</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <p style="margin: 0;">💙 Your results suggest you're managing relatively well, though everyone can benefit from support and self-care strategies.</p>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show counselor recommendation
    counselor_type = recommend_counselor(results)
    recommended_counselor = COUNSELORS[counselor_type]
    
    st.markdown(f"""
    <div class="info-alert">
        <h4 style="margin: 0 0 1rem 0;">🤝 Recommended Support</h4>
        <p style="margin: 0 0 0.5rem 0;"><strong>Based on your assessment, I recommend connecting with {recommended_counselor['name']}:</strong></p>
        <p style="margin: 0.5rem 0;">📞 <strong>Phone:</strong> {recommended_counselor['phone']}</p>
        <p style="margin: 0.5rem 0;">🎯 <strong>Specializes in:</strong> {', '.join(recommended_counselor['expertise'])}</p>
        <p style="margin: 0.5rem 0;">✨ <strong>Best for students with:</strong> {recommended_counselor['focus']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return results

def render_visual_page():
    """Render the visual expression analysis page"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 style="margin: 0;">🎨 Visual Chat Support</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Express your feelings through images, drawings, or journal entries. I'll help you understand what you're communicating.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader(
        "Share your visual expression",
        type=['png', 'jpg', 'jpeg'],
        help="This could be artwork, journal pages, photos that represent your mood, or anything visual you'd like to share"
    )
    
    if uploaded_image:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_image, caption="Your Visual Expression", use_container_width=True)
        
        with col2:
            user_context = st.text_area(
                "Tell me about this image (optional):",
                placeholder="What were you feeling when you created/took this? What does it represent to you?",
                height=100
            )
            
            if st.button("🔍 Analyze Expression", use_container_width=True, type="primary"):
                if 'vision_processor' in st.session_state:
                    with st.spinner("Understanding your visual expression..."):
                        try:
                            # Save uploaded image to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                                temp_img.write(uploaded_image.read())
                                temp_img_path = temp_img.name
                            
                            # Analyze emotional content
                            analysis = st.session_state.vision_processor.analyze_emotional_content(
                                temp_img_path, user_context
                            )
                            
                            # Generate calming audio response
                            audio_path = st.session_state.vision_processor.generate_calming_audio(
                                analysis, use_elevenlabs=True
                            )
                            
                            # Display results
                            st.markdown("""
                            <div class="info-alert">
                                <h4 style="margin: 0 0 1rem 0;">💙 What I See in Your Expression</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(analysis)
                            
                            if audio_path:
                                st.audio(audio_path, format="audio/mp3")
                            
                            # Clean up temp file
                            os.unlink(temp_img_path)
                            
                        except Exception as e:
                            st.error("I'm having trouble analyzing the image right now, but I want you to know that your expression matters. Would you like to tell me about it instead?")
                else:
                    st.error("Image analysis is not available right now, but your creative expression is valuable. Consider sharing your feelings in the chat instead.")

def render_relaxation_page():
    """Render the guided relaxation page"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 style="margin: 0;">🧘 Guided Relaxation & Mindfulness</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Take a few minutes for yourself with these calming exercises</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0;">
            <h3 style="color: #667eea; margin: 0 0 1rem 0;">🫁 Breathing Exercise</h3>
            <p style="color: #666; margin: 0 0 1.5rem 0;">4-7-8 breathing technique for immediate calm</p>
        """, unsafe_allow_html=True)
        
        if st.button("🎧 Start Breathing Exercise", use_container_width=True):
            audio_path = generate_relaxation_audio("breathing", use_elevenlabs=True, elevenlabs_api_key=st.secrets.get("ELEVENLABS_API_KEY"))
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
                st.markdown('<div class="success-alert">💙 Take your time with this exercise. Breathe at your own pace.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0;">
            <h3 style="color: #667eea; margin: 0 0 1rem 0;">🧠 Mindfulness</h3>
            <p style="color: #666; margin: 0 0 1.5rem 0;">5-4-3-2-1 grounding technique</p>
        """, unsafe_allow_html=True)
        
        if st.button("🎧 Start Mindfulness Exercise", use_container_width=True):
            audio_path = generate_relaxation_audio("mindfulness", use_elevenlabs=True, elevenlabs_api_key=st.secrets.get("ELEVENLABS_API_KEY"))
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
                st.markdown('<div class="success-alert">💙 Notice the present moment without judgment.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0;">
            <h3 style="color: #667eea; margin: 0 0 1rem 0;">😌 Progressive Relaxation</h3>
            <p style="color: #666; margin: 0 0 1.5rem 0;">Full body tension release</p>
        """, unsafe_allow_html=True)
        
        if st.button("🎧 Start Relaxation Exercise", use_container_width=True):
            audio_path = generate_relaxation_audio("progressive_relaxation", use_elevenlabs=True, elevenlabs_api_key=st.secrets.get("ELEVENLABS_API_KEY"))
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
                st.markdown('<div class="success-alert">💙 Let your body find its natural state of relaxation.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-alert" style="margin: 2rem 0;">
        <h4 style="margin: 0 0 0.5rem 0;">💡 Tip</h4>
        <p style="margin: 0;">Regular practice of these techniques can help build resilience against stress and anxiety. Even 5 minutes a day can make a difference!</p>
    </div>
    """, unsafe_allow_html=True)
