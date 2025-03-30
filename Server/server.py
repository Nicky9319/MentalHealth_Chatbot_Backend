from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
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

# Load environment variables
load_dotenv()

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

# Get GROQ API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

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

@app.post("/GenerateAudioResponse")
async def generateAudioResponse(request: Request):
    """Saves the audio file sent by the client to the local disk"""
    try:
        # Parse the JSON payload
        payload = await request.json()
        session_id = payload.get("session_id")
        audio_data = payload.get("audio_data")  # This should be the binary data encoded as base64
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID cannot be empty")
        if not audio_data:
            raise HTTPException(status_code=400, detail="Audio data cannot be empty")
        
        # Convert base64 string to binary
        import base64
        audio_binary = base64.b64decode(audio_data)
        
        # Save the audio data to a file in the current script's directory
        current_directory = os.path.dirname(os.path.abspath(__file__))
        audio_file_path = os.path.join(current_directory, "audio.wav")
        
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(audio_binary)
        
        return {"message": "Audio saved successfully", "session_id": session_id, "file_path": audio_file_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)