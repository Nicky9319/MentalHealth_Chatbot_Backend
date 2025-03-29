## Simple Conversation with Chat History using Groq
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

from dotenv import load_dotenv
load_dotenv()

## set up Streamlit 
st.title("Conversational Chat with History")
st.write("Chat with an AI assistant that remembers your conversation")

## Get the Groq API Key from environment variables
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

# Initialize the LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

## chat interface
session_id = st.text_input("Session ID", value="default_session")

## statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Create a simple conversational chain with the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are a psychiatrist who is very kind and helpful and is able to help people out with facing mental issues. 
         There is a person who is suffering from mental illness and you need to assist him or her. 
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
         """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

user_input = st.text_input("Your message:")
if user_input:
    session_history = get_session_history(session_id)
    response = conversational_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    
    # Display chat history
    st.write("Chat History:")
    for message in session_history.messages:
        prefix = "🧑‍💼 You: " if message.type == "human" else "🤖 Assistant: "
        st.write(f"{prefix} {message.content}")




