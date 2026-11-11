
import streamlit as st
from chatbot import PythonChatbot

# Page configuration
st.set_page_config(
    page_title="Python Buddy",
    page_icon="🤖",
    layout="centered"
)

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    return PythonChatbot("python_knowledge.csv")

chatbot = load_chatbot()

# Application heading
st.title("🤖 Python Buddy")
st.subheader("Your Personal Python Question-Answering Chatbot")

st.write(
    "Hello! I am Python Buddy. "
    "Ask me anything related to Python programming."
)

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask a Python question...")

if user_question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate chatbot response
    answer = chatbot.get_answer(user_question)

    # Display chatbot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

st.divider()

st.caption("Python Buddy | NLP and Machine Learning Based Chatbot")
