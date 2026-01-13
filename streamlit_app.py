import tempfile
import streamlit as st
from app import (ingest_documents, load_model,load_embedding_model, create_vector_db,
chunk_documents,retrieve_relevant_docs,get_answer,delete_vector_db,reformulate_query)
from tempfile import NamedTemporaryFile


if "model" not in st.session_state:
    st.session_state["model"] = load_model()
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = load_embedding_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.write("Hello There 😎")

uploaded_file = st.file_uploader("Enter a file","pdf")

arxiv_id  = st.text_input("Enter the arxiv id for your paper")

if not uploaded_file and not arxiv_id:
    st.write("Please upload a file so we can chat")

current_name = uploaded_file.name if uploaded_file else None

if uploaded_file:
    source_type = "uploaded_file"
    source_identifier = uploaded_file
    current_name = uploaded_file.name
elif arxiv_id:
    source_type = "arxiv_id"
    source_identifier = arxiv_id
    current_name = arxiv_id

if uploaded_file or arxiv_id:
    if "paper_name" not in st.session_state or st.session_state["paper_name"] != current_name:
        # clear messages
        st.session_state.messages = []
        
        if "vector_store" in st.session_state:
            delete_vector_db(st.session_state["vector_store"])
            del st.session_state["vector_store"]
        
        try:
            with st.spinner("Processing Documents"):

                chunks,paper_name = ingest_documents(source_type,source_identifier)
                st.session_state["vector_store"] = create_vector_db(chunks,st.session_state["embedding_model"])
                
                st.session_state["paper_name"] = paper_name
                st.success("processed successfully")
            
                
        except Exception as e:
            st.write(f"Error while processing document {e}")
            st.stop()
if "vector_store" in st.session_state:
    prompt = st.chat_input("Say something")
    if prompt:
        query= reformulate_query(st.session_state["messages"],prompt,st.session_state["model"])
        try: 
                    
            relevant_docs = retrieve_relevant_docs(query,st.session_state["vector_store"])

            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role":"user","content":prompt})

            response = get_answer(prompt,st.session_state["model"],relevant_docs)
            with st.chat_message("ai"):
                st.markdown(response)
            st.session_state.messages.append({"role":"ai","content":response})

        except Exception as e:
            st.error(e)

if "paper_name" in st.session_state:
    st.info(f"📄 Current paper: {st.session_state['paper_name']}")
   
