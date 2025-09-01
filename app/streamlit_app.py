import streamlit as st
from app.rag_chain import get_rag_response
st.title("RAG System with DeepSeek R1")
query = st.text_input("Ask a question:")
if query:
	response = get_rag_response(query)
	st.write("### Response:")
	st.write(response)