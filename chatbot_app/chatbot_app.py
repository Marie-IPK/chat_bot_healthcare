import streamlit as st
import chatbot_2  # Assuming your chatbot code is in chatbot_2.py

st.title("Healthcare Chatbot")

user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_2.get_response(chatbot_2.predict_class(user_input), chatbot_2.intents)
    st.write("Bot:", response)