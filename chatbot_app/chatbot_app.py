import streamlit as st
import chat_2  # Assuming your chatbot code is in chat_2.py it's the true name of our app

st.title("Healthcare Chatbot")

user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_2.get_response(chatbot_2.predict_class(user_input), chatbot_2.intents)
    st.write("Bot:", response)