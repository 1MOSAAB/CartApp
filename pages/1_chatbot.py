import streamlit as st
from groq import Groq

st.title("💬 Chatbot")

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def chat_with_groq(user_input):
    if not user_input or not user_input.strip():
        return "Please type a message."

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": user_input}],
        temperature=0.7,
        max_tokens=512
    )
    return response.choices[0].message.content

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    reply = chat_with_groq(user_input)
    st.session_state.chat_history.append(("assistant", reply))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
