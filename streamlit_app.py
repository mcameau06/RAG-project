import streamlit as st
from app import ingest_documents
from tempfile import NamedTemporaryFile
st.write("Hello There 😎")

uploaded_file = st.file_uploader("Enter a file","pdf")

if uploaded_file:
    with NamedTemporaryFile(delete=False,suffix="pdf") as temp:
        temp.write(uploaded_file.getvalue())
        pdf_path = temp.name

    paper = ingest_documents(pdf_path)
    
   
