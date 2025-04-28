import os
import streamlit as st
from groq import Groq

# Set your Groq API key
API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Function to send input to Groq
def chat_with_groq(user_input):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("💬Chatbot")

# Initialize chat history in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box at the bottom
user_input = st.chat_input("Say something...")

if user_input:
    # Display user message
    st.session_state.chat_history.append(("user", user_input))
    # Get Groq's response
    response = chat_with_groq(user_input)
    st.session_state.chat_history.append(("bot", response))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)