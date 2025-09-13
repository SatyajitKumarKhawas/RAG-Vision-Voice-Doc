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

# Voice input HTML/JS component
def create_voice_input_component():
    """Create the voice input HTML component"""
    voice_html = """
    <div style="padding: 10px; border: 2px dashed #ccc; border-radius: 10px; margin: 10px 0; text-align: center;">
        <h4 style="margin-top: 0;">🎤 Voice Input</h4>
        <button id="startBtn" onclick="startRecording()" style="
            background-color: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">🎤 Start Recording</button>
        
        <button id="stopBtn" onclick="stopRecording()" disabled style="
            background-color: #f44336; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">🛑 Stop Recording</button>
        
        <div id="status" style="margin: 10px; font-weight: bold; color: #666;"></div>
        <div id="transcript" style="
            margin: 10px; 
            padding: 10px; 
            background-color: #f0f0f0; 
            border-radius: 5px; 
            min-height: 40px;
            font-style: italic;
        ">Your transcribed text will appear here...</div>
        
        <button id="sendBtn" onclick="sendToChat()" disabled style="
            background-color: #2196F3; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">📤 Send to Chat</button>
        
        <button id="clearBtn" onclick="clearTranscript()" style="
            background-color: #ff9800; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">🗑 Clear</button>
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
            document.getElementById('status').style.color = '#f44336';
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
            document.getElementById('status').style.color = '#f44336';
            resetButtons();
        };

        recognition.onend = function() {
            isRecording = false;
            resetButtons();
            if (finalTranscript.trim() !== '') {
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('status').innerHTML = '✅ Recording completed!';
                document.getElementById('status').style.color = '#4CAF50';
            } else {
                document.getElementById('status').innerHTML = '⚠ No speech detected';
                document.getElementById('status').style.color = '#ff9800';
            }
        };
    } else {
        document.getElementById('status').innerHTML = '❌ Speech recognition not supported in this browser';
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
            // Store in session storage for Streamlit to pick up
            parent.sessionStorage.setItem('voice_input', finalTranscript.trim());
            
            document.getElementById('status').innerHTML = '📤 Sent to chat!';
            document.getElementById('status').style.color = '#4CAF50';
        }
    }

    function clearTranscript() {
        finalTranscript = '';
        document.getElementById('transcript').innerHTML = 'Your transcribed text will appear here...';
        document.getElementById('sendBtn').disabled = true;
        document.getElementById('status').innerHTML = '';
        parent.sessionStorage.removeItem('voice_input');
    }
    </script>
    """
    return voice_html

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
        slow=slow_speed  # Slower for more calming effect
    )
    audioobj.save(output_filepath)
    return output_filepath

def text_to_speech_with_elevenlabs(input_text, output_filepath, api_key):
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=api_key)

        # Try multiple voice options in order of preference
        voice_options = ["Rachel", "Aria", "Sarah", "Nicole", "Bella"]
        
        for voice_id in voice_options:
            try:
                response = client.text_to_speech.convert(
                    voice_id=voice_id,
                    model_id="eleven_turbo_v2",
                    text=input_text
                )

                with open(output_filepath, "wb") as f:
                    for chunk in response:
                        f.write(chunk)

                return output_filepath
            except Exception:
                continue  # Try next voice
        
        # If all voices fail, fallback to gTTS silently
        return text_to_speech_with_gtts(input_text, output_filepath)
        
    except Exception:
        # Silent fallback to gTTS
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
    # For GHQ-12, responses are scored differently
    # Better/Same = 0, Less/Much less = 1
    binary_responses = []
    for resp in responses:
        if resp in [0, 1]:  # Better than usual, Same as usual
            binary_responses.append(0)
        else:  # Less than usual, Much less than usual
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
    
    # Default to general counselor
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
        
        # Create vectorstore directory if it doesn't exist
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
            model_name = getattr(st.session_state, 'selected_model', 'gemini-2.0-flash')
            google_api_key = st.secrets.get("GOOGLE_API_KEY")

            if not google_api_key:
                st.error("Google API Key not found in Streamlit Secrets.")
                return

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.3,  # Slightly more creative for counseling
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
            return None

def create_screening_interface(test_type):
    """Create interface for mental health screening tests"""
    
    if test_type == "PHQ-9":
        questions = PHQ9_QUESTIONS
        options = RESPONSE_OPTIONS
        st.subheader("📋 PHQ-9 Depression Screening")
        st.write("*This questionnaire helps assess depression symptoms over the past 2 weeks.*")
        
    elif test_type == "GAD-7":
        questions = GAD7_QUESTIONS
        options = RESPONSE_OPTIONS
        st.subheader("📋 GAD-7 Anxiety Screening")
        st.write("*This questionnaire helps assess anxiety symptoms over the past 2 weeks.*")
        
    elif test_type == "GHQ-12":
        questions = GHQ12_QUESTIONS
        options = GHQ_RESPONSE_OPTIONS
        st.subheader("📋 GHQ-12 General Health Screening")
        st.write("*This questionnaire assesses your recent psychological wellbeing.*")
    
    responses = []
    
    with st.form(f"{test_type}_form"):
        st.write("*Please answer honestly. Your responses are confidential and will help us understand how to support you better.*")
        
        for i, question in enumerate(questions):
            response = st.radio(
                question,
                options=list(range(len(options))),
                format_func=lambda x: options[x],
                key=f"{test_type}_{i}",
                horizontal=True
            )
            responses.append(response)
        
        submitted = st.form_submit_button("Submit Assessment")
        
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

def display_assessment_results():
    """Display assessment results with empathetic interpretation"""
    results = {}
    
    # Collect all assessment results
    for test in ['phq-9', 'gad-7', 'ghq-12']:
        if f'{test}_results' in st.session_state:
            results[test.replace('-', '')] = st.session_state[f'{test}_results']
    
    if not results:
        return None
    
    st.subheader("📊 Your Assessment Results")
    
    # Display results with empathetic framing
    for test_name, result in results.items():
        score = result['score']
        severity = result['severity']
        
        st.write(f"**{test_name.upper()} Results:**")
        st.write(f"Score: {score} - {severity}")
        
        # Provide empathetic interpretation
        if "severe" in severity.lower():
            st.warning("💙 Your results suggest you might be experiencing significant distress. Please know that you're not alone, and seeking support is a sign of strength. I'd recommend speaking with one of our professional counselors.")
        elif "moderate" in severity.lower():
            st.info("💙 Your results indicate you might be experiencing some challenges. It's completely normal to go through difficult times, and there are effective ways to help you feel better.")
        else:
            st.success("💙 Your results suggest you're managing relatively well, though everyone can benefit from support and self-care strategies.")
        
        st.markdown("---")
    
    return results

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
        return None

def check_voice_input():
    """Check for voice input from session storage"""
    voice_input_js = """
    <script>
    const voiceInput = sessionStorage.getItem('voice_input');
    if (voiceInput) {
        sessionStorage.removeItem('voice_input');
        return voiceInput;
    }
    return null;
    </script>
    """
    return components.html(voice_input_js, height=0)

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

def main():
    st.set_page_config(
        page_title="Student Mental Health AI Counselor",
        page_icon="🧠💙",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for ChatGPT-like styling
    st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        max-width: 850px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1f1f1f;
    }
    
    .main-header p {
        color: #666;
        font-size: 1rem;
        margin: 0;
    }
    
    /* Quick support buttons styling */
    .quick-support {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin: 1rem 0 2rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 12px;
    }
    
    .stButton > button {
        background: white;
        border: 1px solid #d0d7de;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        color: #24292f;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #f6f8fa;
        border-color: #8c959f;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: white;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #e0e0e0;
    }
    
    /* Input styling */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 1px solid #d0d7de;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f6f8fa;
        border-radius: 8px;
        border: 1px solid #d0d7de;
    }
    
    /* Voice input container */
    .voice-container {
        background: white;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        color: #666;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white;
        color: #1f1f1f;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠💙 Mental Health AI Counselor</h1>
        <p>A safe space for emotional support, guidance, and mental health resources</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙ Configuration & Resources")
        
        # API Keys
        st.subheader("🔑 API Keys")
        google_api_key = st.secrets.get("GOOGLE_API_KEY")
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")
        
        st.text_input(
            "Google API Key (Gemini):",
            value="Set in Streamlit Secrets" if google_api_key else "",
            type="password",
            disabled=True
        )
        
        st.text_input(
            "GROQ API Key:",
            value="Set in Streamlit Secrets" if groq_api_key else "",
            type="password",
            disabled=True
        )
        
        st.text_input(
            "ElevenLabs API Key (Optional):",
            value="Set in Streamlit Secrets" if elevenlabs_api_key else "",
            type="password",
            disabled=True
        )
        
        use_elevenlabs = st.checkbox(
            "Use ElevenLabs TTS", 
            value=bool(elevenlabs_api_key),
            help="Uncheck to use free gTTS instead"
        )
        
        st.markdown("---")
        
        # Mental Health Resources
        st.subheader("🧠 Mental Health Database")
        vectorstore_exists = os.path.exists(DB_FAISS_PATH)
        
        if vectorstore_exists:
            st.success("✅ Mental health knowledge base loaded!")
        else:
            st.warning("⚠ Please load mental health resources (Gale Encyclopedia of Mental Health PDFs)")
        
        if st.button("📚 Process Mental Health PDFs"):
            with st.spinner("Processing mental health resources..."):
                documents = DocumentProcessor.load_pdf_files(DATA_PATH)
                
                if not documents:
                    st.error("No PDF files found! Please add Gale Encyclopedia of Mental Health PDFs to the data directory.")
                else:
                    text_chunks = DocumentProcessor.create_chunks(documents)
                    DocumentProcessor.create_vectorstore(text_chunks)
                    st.rerun()
        
        # Model selection
        model_options = {
            "Gemini 2.0 Flash (Recommended)": "gemini-2.0-flash", 
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Pro": "gemini-2.5-pro"
        }
        
        selected_model = st.selectbox(
            "Choose Gemini Model:",
            options=list(model_options.keys())
        )
        
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_options[selected_model]
        
        if st.session_state.selected_model != model_options[selected_model]:
            st.session_state.selected_model = model_options[selected_model]
            if 'counselor' in st.session_state:
                del st.session_state.counselor
        
        st.markdown("---")
        
        # Campus Counselors
        st.subheader("👥 Campus Counselors")
        st.markdown("*Our professional counselors are here for you:*")
        
        for key, counselor in COUNSELORS.items():
            with st.expander(f"🧑‍⚕️ {counselor['name']}"):
                st.write(f"**📞 Phone:** {counselor['phone']}")
                st.write(f"**🎓 Qualification:** {counselor['qualification']}")
                st.write(f"**🔍 Expertise:** {', '.join(counselor['expertise'])}")
                st.write(f"**💬 Style:** {counselor['style']}")
                st.write(f"**🎯 Best for:** {counselor['focus']}")
        
        st.markdown("---")
        st.subheader("ℹ How This Works")
        st.markdown("""
        **🎯 Smart Mental Health Support:**
        - **Chat/Voice** → AI Counselor (empathetic responses)
        - **Images** → Emotional content analysis
        - **Assessments** → Standardized mental health screening
        - **Crisis Detection** → Immediate professional referral
        
        **🔒 Your privacy matters:**
        - Confidential conversations
        - No data stored permanently
        - Professional referrals when needed
        """)
    
    # Check API keys
    if not google_api_key:
        st.error("Please provide your Google API key in Streamlit secrets for the counselor functionality.")
    
    if not groq_api_key:
        st.error("Please provide your GROQ API key in Streamlit secrets for image analysis.")
    
    # Main interface - single column ChatGPT style
    
    # Quick support buttons
    st.markdown('<div class="quick-support">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("😰 I'm feeling anxious", key="anxiety_btn"):
            st.session_state.messages.append({
                'role': 'user',
                'content': "I'm feeling really anxious right now and I don't know what to do."
            })
            st.rerun()
    
    with col2:
        if st.button("😔 I'm feeling down", key="down_btn"):
            st.session_state.messages.append({
                'role': 'user',
                'content': "I've been feeling really down lately and nothing seems to help."
            })
            st.rerun()
    
    with col3:
        if st.button("😵 I'm overwhelmed", key="overwhelmed_btn"):
            st.session_state.messages.append({
                'role': 'user',
                'content': "I'm feeling completely overwhelmed with everything going on."
            })
            st.rerun()
    
    with col4:
        if st.button("😟 Need to talk", key="talk_btn"):
            st.session_state.messages.append({
                'role': 'user',
                'content': "I just need someone to talk to right now. I'm not feeling great."
            })
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Voice input in a cleaner container
    with st.container():
        st.markdown('<div class="voice-container">', unsafe_allow_html=True)
        st.markdown("**🎤 Voice Input**")
        components.html(create_voice_input_component(), height=250)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional features in tabs (but more subtle)
    with st.expander("📋 Additional Features", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Mental Health Screening", "Guided Relaxation", "Visual Expression"])
        
        with tab1:
            st.write("*Confidential assessments to help understand your mental health better*")
            screening_type = st.selectbox(
                "Choose assessment:",
                ["Select...", "PHQ-9 (Depression)", "GAD-7 (Anxiety)", "GHQ-12 (General Mental Health)"],
                key="screening_select"
            )
            
            if screening_type != "Select...":
                test_name = screening_type.split(" ")[0]
                create_screening_interface(test_name)
            
            # Display results if available
            results = display_assessment_results()
        
        with tab2:
            st.write("*Take a few minutes for yourself with calming exercises*")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🫁 Breathing Exercise", key="breathing_btn"):
                    audio_path = generate_relaxation_audio("breathing", use_elevenlabs, elevenlabs_api_key)
                    if audio_path:
                        st.audio(audio_path, format="audio/mp3")
                        st.success("💙 Take your time with this exercise")
            
            with col2:
                if st.button("🧠 Mindfulness", key="mindfulness_btn"):
                    audio_path = generate_relaxation_audio("mindfulness", use_elevenlabs, elevenlabs_api_key)
                    if audio_path:
                        st.audio(audio_path, format="audio/mp3")
                        st.success("💙 Notice the present moment")
            
            with col3:
                if st.button("😌 Progressive Relaxation", key="relaxation_btn"):
                    audio_path = generate_relaxation_audio("progressive_relaxation", use_elevenlabs, elevenlabs_api_key)
                    if audio_path:
                        st.audio(audio_path, format="audio/mp3")
                        st.success("💙 Let your body relax")
        
        with tab3:
            st.write("*Share visual expressions of your feelings*")
            
            uploaded_image = st.file_uploader(
                "Upload image (artwork, journal, photos)",
                type=['png', 'jpg', 'jpeg'],
                key="visual_upload"
            )
            
            if uploaded_image:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(uploaded_image, caption="Your Expression", use_container_width=True)
                
                with col2:
                    user_context = st.text_area(
                        "Tell me about this (optional):",
                        height=100,
                        key="visual_context"
                    )
                    
                    if st.button("🔍 Analyze", key="analyze_btn"):
                        if 'vision_processor' in st.session_state:
                            with st.spinner("Understanding your expression..."):
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                                        temp_img.write(uploaded_image.read())
                                        temp_img_path = temp_img.name
                                    
                                    analysis = st.session_state.vision_processor.analyze_emotional_content(
                                        temp_img_path, user_context
                                    )
                                    
                                    st.markdown("**💙 What I see in your expression:**")
                                    st.markdown(analysis)
                                    
                                    # Add to main chat
                                    st.session_state.messages.append({
                                        'role': 'user',
                                        'content': f"📸 Shared visual expression" + (f": {user_context}" if user_context else ""),
                                    })
                                    
                                    st.session_state.messages.append({
                                        'role': 'assistant',
                                        'content': analysis,
                                        'type': 'emotional_analysis'
                                    })
                                    
                                    os.unlink(temp_img_path)
                                    
                                except Exception:
                                    st.error("Having trouble with the image. Would you like to describe it instead?")
    
    # Initialize components
    if google_api_key and 'counselor' not in st.session_state:
        if vectorstore_exists:
            st.session_state.counselor = MentalHealthCounselor()
    
    if groq_api_key and 'vision_processor' not in st.session_state:
        st.session_state.vision_processor = VisionProcessor(groq_api_key, elevenlabs_api_key)
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat messages container
    st.markdown("### 💬 Conversation")
    
    # Display chat messages in ChatGPT style
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message.get('type') == 'crisis_response':
                st.error("🚨 **Crisis Support Needed**")
            elif message.get('type') == 'counselor_recommendation':
                st.info("👥 **Counselor Recommendation**")
            elif message.get('type') == 'emotional_analysis':
                st.info("🖼 **Emotional Expression Analysis**")
            
            st.markdown(message['content'])
            
            # Display audio if available
            if message.get('audio_path') and os.path.exists(message['audio_path']):
                st.audio(message['audio_path'], format="audio/mp3")
            
            # Display source documents if available
            if message.get('source_docs') and message['role'] == 'assistant':
                with st.expander("📚 Mental Health Knowledge Sources"):
                    for i, doc in enumerate(message['source_docs'], 1):
                        st.markdown(f"**Source {i}:**")
                        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        st.text(content_preview)
                        
                        if hasattr(doc, 'metadata') and doc.metadata:
                            metadata = doc.metadata
                            st.json({
                                "source": metadata.get('source', 'Unknown'),
                                "page": metadata.get('page', 'Unknown'),
                                "page_label": metadata.get('page_label', 'Unknown'),
                                "total_pages": metadata.get('total_pages', 'Unknown')
                            })
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Share what's on your mind... I'm here to listen 💙"):
        # Check for crisis language first
        if detect_crisis_language(prompt):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            
            crisis_response = generate_crisis_response()
            st.session_state.messages.append({
                'role': 'assistant',
                'content': crisis_response,
                'type': 'crisis_response'
            })
            st.rerun()
        
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Process with Mental Health Counselor
        if 'counselor' in st.session_state and vectorstore_exists:
            with st.chat_message('assistant'):
                with st.spinner("Listening and reflecting on what you've shared..."):
                    result, source_docs = st.session_state.counselor.get_response(prompt)
                    
                    st.markdown(result)
                    
                    # Display source documents if available
                    if source_docs:
                        with st.expander("📚 Mental Health Knowledge Sources"):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                # Show content preview
                                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                st.text(content_preview)
                                
                                # Show metadata in formatted way
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    metadata = doc.metadata
                                    st.json({
                                        "source": metadata.get('source', 'Unknown'),
                                        "page": metadata.get('page', 'Unknown'),
                                        "page_label": metadata.get('page_label', 'Unknown'),
                                        "total_pages": metadata.get('total_pages', 'Unknown')
                                    })
                                st.markdown("---")
                    
                    # Generate calming audio response
                    audio_path = None
                    if 'vision_processor' in st.session_state:
                        audio_path = st.session_state.vision_processor.generate_calming_audio(
                            result, use_elevenlabs
                        )
                        if audio_path:
                            st.audio(audio_path, format="audio/mp3")
                    
                    # Check if counselor recommendation is needed
                    assessment_results = {}
                    for test in ['phq-9', 'gad-7', 'ghq-12']:
                        if f'{test}_results' in st.session_state:
                            assessment_results[test.replace('-', '')] = st.session_state[f'{test}_results']
                    
                    counselor_type = recommend_counselor(assessment_results, prompt)
                    recommended_counselor = COUNSELORS[counselor_type]
                    
                    # Show counselor recommendation
                    with st.expander("🤝 Professional Support Available"):
                        st.markdown(f"**Based on what you've shared, I think {recommended_counselor['name']} might be a great fit for you:**")
                        st.markdown(f"📞 **Phone:** {recommended_counselor['phone']}")
                        st.markdown(f"🎯 **They specialize in:** {', '.join(recommended_counselor['expertise'])}")
                        st.markdown(f"💭 **Their approach:** {recommended_counselor['style']}")
                        st.markdown(f"✨ **Best for:** {recommended_counselor['focus']}")
                    
                    # Prepare content with sources for session state
                    content_with_sources = result
                    if source_docs:
                        content_with_sources += f"\n\n*Sources:* {len(source_docs)} mental health resource(s) referenced"
                    
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': content_with_sources,
                        'type': 'counselor_response',
                        'audio_path': audio_path,
                        'recommended_counselor': recommended_counselor,
                        'source_docs': source_docs
                    })
        else:
            st.error("The counselor support system isn't available right now, but your feelings are valid. Consider reaching out to one of our campus counselors directly.")
    
    # Footer with crisis resources (ChatGPT-style)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem 0; font-size: 0.9rem; line-height: 1.6;'>
        <strong>💙 You're not alone.</strong> This AI provides support, but professional counselors are available for deeper help.<br>
        <strong>🔒 Confidential:</strong> Conversations are private and not stored permanently.<br>
        <strong>🚨 Crisis Support:</strong> Call <strong>112 (Emergency)</strong> or <strong>1860-2662-345 (iCALL)</strong> immediately if you're in crisis.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

