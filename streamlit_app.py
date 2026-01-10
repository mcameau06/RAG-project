import streamlit as st
from app import ingest_documents, load_model,load_embedding_model, create_vector_db,chunk_documents,retrieve_relevant_docs,get_answer
from tempfile import NamedTemporaryFile
st.write("Hello There 😎")

uploaded_file = st.file_uploader("Enter a file","pdf")
if "model" not in st.session_state:
    st.session_state["model"] = load_model()
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = load_embedding_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if uploaded_file:
    with NamedTemporaryFile(delete=False,suffix="pdf") as temp:
        temp.write(uploaded_file.getvalue())
        pdf_path = temp.name
        if "paper_name" not in st.session_state:
            st.session_state["paper_name"] = uploaded_file.name

    try:

        paper = ingest_documents(pdf_path)

        chunks = chunk_documents(paper)
        if "vector_store" not in st.session_state:
                st.session_state["vector_store"] = create_vector_db(chunks,st.session_state["embedding_model"])
                st.success("processed successfully")
        
    except Exception as e:
        st.write(e)

prompt = st.chat_input("Say something")
if prompt:
    
    try: 
        relevant_docs = retrieve_relevant_docs(st.session_state["embedding_model"],prompt)
        response = get_answer(prompt,st.session_state["model"],relevant_docs)

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})

        response = get_answer(prompt,st.session_state["model"],relevant_docs)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role":"assistant","content":response})

    except Exception as e:
        st.error(e)


   
