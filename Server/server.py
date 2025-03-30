from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware  # Add this import

import speech_recognition as sr
from pathlib import Path
import os
from groq import Groq
import json
import librosa
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get GROQ API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

class VoiceAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.client = Groq(api_key=api_key)  # Fixed: Use api_key directly, not os.getenv(api_key))

    def extract_audio_features(self, audio_file_path):
        """Extract audio features like pitch, volume, and speaking rate"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path)
            
            # Extract features
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/2])
            
            # Volume (RMS energy)
            volume = np.mean(librosa.feature.rms(y=y))
            
            # Speaking rate (zero crossing rate)
            speaking_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            return {
                "pitch": float(pitch_mean),
                "volume": float(volume),
                "speaking_rate": float(speaking_rate)
            }
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return None

    def analyze_emotion(self, audio_file_path):
        """Analyze emotional factors from voice tonality and transcribed text"""
        # Get text transcription
        text = self.transcribe_audio(audio_file_path)
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio_file_path)
        
        if not audio_features:
            return None

        prompt = f"""
        Analyze the following voice characteristics and text for emotional indicators:
        
        Text transcription: {text}
        Voice features:
        - Pitch: {audio_features['pitch']} (higher values indicate higher pitch)
        - Volume: {audio_features['volume']} (higher values indicate louder speech)
        - Speaking rate: {audio_features['speaking_rate']} (higher values indicate faster speech)
        
        Please provide:
        1. Stress level (1-10)
        2. Confidence level (1-10)
        3. Anxiety level (1-10)
        4. Overall mood (positive/negative/neutral)
        5. Key emotional indicators based on both voice and text
        
        """

        response = self.client.chat.completions.create(
            model="Gemma2-9b-It",
            messages=[
                {"role": "system", "content": "You are an expert in voice emotion analysis."},
                {"role": "user", "content": prompt}
            ]
        )

        # print(response)
        # print()
        # print(response.choices[0].message.content)

        # print(type(response.choices[0].message.content))

        try:
            emotion_data = response.choices[0].message.content
            print(type(emotion_data))
            # Include the transcription in the return value
            # emotion_data["transcription"] = text
            return emotion_data
        except Exception as e:
            print(e)
            return {
                "stress_level": 0,
                "confidence_level": 0,
                "anxiety_level": 0,
                "mood": "unknown",
                "emotional_indicators": [],
                "transcription": text
            }


    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribes an audio file using the Groq client.

        Args:
            audio_file_path (str): Path to the audio file.
            model (str): Model to use for transcription. Default is "whisper-large-v3-turbo".
            prompt (str): Optional context or spelling prompt.
            response_format (str): Format of the response. Default is "verbose_json".
            timestamp_granularities (list): Optional list of timestamp granularities ("word", "segment", or both).
            temperature (float): Optional temperature setting for transcription. Default is 0.0.

        Returns:
            str: Transcription text.
        """
        # Initialize the Groq client

        model="whisper-large-v3-turbo"
        prompt=None 
        response_format="verbose_json" 
        timestamp_granularities=None 
        temperature=0.0


        client = Groq()

        # Open the audio file
        with open(audio_file_path, "rb") as file:
            # Create a transcription of the audio file
            transcription = client.audio.transcriptions.create(
                file=file,
                model=model,
                prompt=prompt,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
                temperature=temperature
            )
            # Return the transcription text
            return transcription.text
    
    def process_audio(self, audio_path):
                """Process audio file and return comprehensive analysis"""
                if not os.path.exists(audio_path):
                    return {"error": "Audio file not found"}
                    
                try:
                    emotion_analysis = self.analyze_emotion(audio_path)
                    audio_features = self.extract_audio_features(audio_path)
                    transcription = self.transcribe_audio(audio_path)
                    
                    response = {
                        "emotion_analysis": emotion_analysis,
                        "audio_features": audio_features,
                        "transcription": transcription
                    }
                    print("Final Analysis:", json.dumps(response, indent=2))
                    return response
                except Exception as e:
                    return {"error": str(e)}
                    

# Initialize the voice analyzer
voice_analyzer = VoiceAnalyzer()

# Initialize FastAPI app
app = FastAPI(title="Mental Health QA Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, replace with specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

# Session counter for sequential session IDs
session_counter = 1

# Store for chat histories
chat_store: Dict[str, ChatMessageHistory] = {}

# Changed from Dict to single string for global shared context
global_context = ""

# Pydantic models for request/response
class Message(BaseModel):
    content: str
    type: str  # "human" or "ai"

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SessionHistory(BaseModel):
    session_id: str
    messages: List[Message]

# Add new model for audio response
class AudioResponse(BaseModel):
    message: str
    session_id: str
    transcription: str
    response: str
    emotion_analysis: Any
    audio_features: Dict[str, float]

# Function to update global context with new information from all users
def update_global_context(user_input: str, session_id: str) -> str:
    global global_context
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are analyzing a conversation with a mental health patient.
        Extract key personal information from the user's message that might be relevant for future sessions.
        Focus on 
        demographic information, 
        mental health indicators, 
        lifestyle details, 
        medical conditions,
        Family Members name and their personal information
        and other context that gives the information about the person itself in whatsoever manner possible.
        
        If the existing context contains information, update it with any new details or correct any inconsistencies.
        Provide a concise, well-organized summary that preserves all important information.
        
        Existing context:
        {existing_context}
        """),
        ("human", f"New message from session {session_id}: {user_input}")
    ])
    
    # Use the single global context variable instead of session-specific context
    context_chain = context_prompt | llm
    updated_context = context_chain.invoke({"existing_context": global_context}).content
    global_context = updated_context
    return updated_context

# Create the updated system prompt template that includes global context
SYSTEM_PROMPT = """
You are a psychiatrist who is very kind and helpful and is able to help people out with facing mental issues. 
There is a person who is suffering from mental illness and you need to assist him or her. 

GLOBAL CONTEXT (Information shared across sessions):
{global_context}

Try finding out the following things about the patient before giving consultation:
1. Gender 
2. Age
3. Sleeping Schedule and Quality
4. Physical Excercise
5. Occupation
6. Financial Condition
7. Social Interaction

Now whenever the patient gives a message try finding out the stress and confidence level of the Person itself so that it can be later used to review the prospect and give advice accordingly

Make the Responses User friendly and not in a manner that seems like it is trying to force out on answer from the patient itself and try to be as helpful as possible.
Be Polite, humble and be more conversational and in between the conversations figure out whetheer it is right time to pose a question and get the response understand the conversation 
context beforehand to make that decision

After you belive you have all the information needed from the patient itself then start moving onto to give the responsees and consulatation to the patient itself.
Keep in mind that you also have to keep into considerations the demographics of the patient itself and also the stress and confidence level of the patient itself.

Okay so give the response while keeping these thinsgs in mind
1. You need to compliment them for opening up from time to time so that they get engaged and feel safe
2. use real life examples to showcase that they are not alone and vulnerable in this position there were different people who had the same scenario and they made out
3. You need to keep the conversation engaging so a counter question regarding the context could be useful
4. Make sure that you dont get lost in the conversation itself in the end the main motive is to provide the consulation for the current scenario and the solution for their problem
5. try to get as much information as possible from the patient about the current situation he or she is in so that we can analyse it better
6. Make sure that you are not giving any medical advice and you are just a psychiatrist who is helping the patient out with his or her problems and not a doctor who is giving out medical advice
7. Start dropping hints in between telling how much severe the case is and when you feel like u know everything then only give a crystal clear decision on what further steps he  or she should take
8. Be Affirmative with your thoughts and take the command in the conversation itself, dont be on the back seat
9. dont make unncesaary comments cause that could lead to the patient getting offended and not opening up to you
10. At any point if the conversation tends towards illegal matters or harming other people, the conversation needs to start tending towards a very decisive end and the next steps he should take, 
11. if the matter escalates bail out and mention the limitations u have as a chatbot itself.
12. No Knowledge or information or question should be asked on the procedure of how a illegal or immoral activity is done or how to do it, if the patient asks that then you need to bail out and mention that you are not allowed to do that and you are not a doctor who is giving out medical advice
"""

# Function to get or create chat history
def getSessionHistory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

# Initialize conversation chain with updated prompt that includes global context
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

conversational_chain = RunnableWithMessageHistory(
    chain,
    getSessionHistory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Endpoints
@app.get("/FetchAllSessions", response_model=List[str])
def getAllSessions():
    """Returns a list of all available session IDs"""
    print("API CALL")
    return list(chat_store.keys())

@app.get("/FetchSessionMessageHistory/{session_id}", response_model=SessionHistory)
def getSessionMessageHistory(session_id: str):
    """Returns the chat history for a specific session"""
    if session_id not in chat_store:
        raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found")
    
    history = getSessionHistory(session_id)
    formatted_messages = []
    
    for msg in history.messages:
        msg_type = "human" if isinstance(msg, HumanMessage) else "ai"
        formatted_messages.append(Message(content=msg.content, type=msg_type))
    
    # Ensure we're returning exactly what the frontend expects
    return SessionHistory(session_id=session_id, messages=formatted_messages)

@app.post("/GenerateResponse", response_model=ChatResponse)
async def generateResponse(request: Request):
    """Generates a response for a given user message and session, with global context"""

    print("Generate Response API CALL")

    # Parse the JSON payload
    payload = await request.json()
    session_id = payload.get("session_id")
    message = payload.get("message")

    print(f"Session ID: {session_id}, Message: {message}")

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty")
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Update the shared global context with the new user message
    updated_context = update_global_context(message, session_id)
    
    # Generate response with the shared global context included
    response = conversational_chain.invoke(
        {
            "input": message,
            "global_context": global_context  # Use the shared global context
        },
        config={
            "configurable": {"session_id": session_id}
        }
    )
    
    return ChatResponse(response=response.content, session_id=session_id)

# Function to get the next session ID - modified to return sequential numbers
def get_next_session_id() -> str:
    global session_counter
    session_id = str(session_counter)  # Simple numeric ID as string
    session_counter += 1
    return session_id

# Create a new session - modified to only use auto-generated IDs
@app.post("/CreateSession", response_model=dict)
def createSession():
    """Creates a new chat session with a sequential numeric ID"""
    print("Create Session API CALL")
    
    session_id = get_next_session_id()
    getSessionHistory(session_id)  # This will create the session
    
    return {"message": f"Session {session_id} created successfully", "session_id": session_id}

# Modified endpoint to maintain backward compatibility but redirect to the main method
@app.post("/CreateSession/{session_id}")
def createSessionWithId(session_id: str):
    """Legacy endpoint - creates a new session with auto-generated ID regardless of input"""
    # Ignore the provided session_id and create a new one with sequential ID
    result = createSession()
    return result

# Update existing generateAudioResponse endpoint

# @app.post("/GenerateAudioResponse", response_model=AudioResponse)
# async def generateAudioResponse(request: Request):
#     """Processes audio and generates a response based on transcription and analysis"""
#     try:
#         # Parse the JSON payload
#         payload = await request.json()
#         session_id = payload.get("session_id")
#         audio_data = payload.get("audio_data")  # This should be the binary data encoded as base64
        
#         if not session_id:
#             raise HTTPException(status_code=400, detail="Session ID cannot be empty")
#         if not audio_data:
#             raise HTTPException(status_code=400, detail="Audio data cannot be empty")
        
#         # Convert base64 string to binary
#         import base64
#         audio_binary = base64.b64decode(audio_data)
        
#         # Save the audio data to a file in the current script's directory
#         current_directory = os.path.dirname(os.path.abspath(__file__))
#         audio_file_path = os.path.join(current_directory, f"audio_{session_id}.wav")
        
#         with open(audio_file_path, "wb") as audio_file:
#             audio_file.write(audio_binary)
        
#         # Process the audio file
#         analysis_result = voice_analyzer.process_audio(audio_file_path)
        
#         if "error" in analysis_result:
#             raise HTTPException(status_code=500, detail=analysis_result["error"])
        
#         # Extract transcription from analysis
#         transcription = analysis_result["transcription"]
        
#         # Update the shared global context with the transcription
#         updated_context = update_global_context(transcription, session_id)
        
#         # Generate response using the transcription as input
#         response = conversational_chain.invoke(
#             {
#                 "input": transcription,
#                 "global_context": global_context  # Use the shared global context
#             },
#             config={
#                 "configurable": {"session_id": session_id}
#             }
#         )
        
#         return AudioResponse(
#             message="Audio processed successfully",
#             session_id=session_id,
#             transcription=transcription,
#             response=response.content,
#             emotion_analysis=analysis_result["emotion_analysis"],
#             audio_features=analysis_result["audio_features"]
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Add a new endpoint to just get audio analysis without generating a response
@app.post("/GenerateAudioResponse")
async def analyzeAudio(request: Request):
    """Analyzes audio without generating a conversational response"""
    try:
        # Parse the JSON payload
        payload = await request.json()
        session_id = payload.get("session_id")
        audio_data = payload.get("audio_data")
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Audio data cannot be empty")
        
        # Convert base64 string to binary
        import base64
        audio_binary = base64.b64decode(audio_data)
        
        # Save the audio data to a file
        current_directory = os.path.dirname(os.path.abspath(__file__))
        audio_file_path = os.path.join(current_directory, f"analysis_{session_id}.wav")
        
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(audio_binary)
        
        # Process the audio file
        analysis_result = voice_analyzer.process_audio(audio_file_path)
        
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result["error"])
        
        return analysis_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
