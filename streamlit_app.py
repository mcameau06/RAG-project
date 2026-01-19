import streamlit as st
import logging
from app import (ingest_documents, load_model,load_embedding_model, create_vector_db,
retrieve_relevant_docs,get_answer,delete_vector_db,reformulate_query)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



if "model" not in st.session_state:
    logger.info("Loading model into session state...")
    st.session_state["model"] = load_model()
if "embedding_model" not in st.session_state:
    logger.info("Loading embedding model into session state...")
    st.session_state["embedding_model"] = load_embedding_model()
if "paper_id" not in st.session_state:
    st.session_state["paper_id"] = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_document():
    logger.info("Resetting document state...")
    for key in ["vector_store", "paper_id", "paper_name", "messages"]:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("Document state reset complete")
    

st.header("Research GPT", text_alignment="center",width="stretch",divider="gray")
st.sidebar.subheader("Chat with your favorite or most confusing research papers!",divider="gray")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


uploaded_file = None
arxiv_id = None

if "vector_store" not in st.session_state:
    with st.chat_message("ai"):
        st.markdown("Please enter an ArXiv ID or upload a document to get started.")
    arxiv_id = st.sidebar.text_input("You can enter the Arxiv id for your paper")
    uploaded_file =  st.sidebar.file_uploader("Or insert a file","pdf")
    
else:
     st.sidebar.button("New Paper/New Chat",on_click=reset_document)

    

if uploaded_file:
    source_type = "uploaded_file"
    source_identifier = uploaded_file
    st.session_state["paper_id"] = uploaded_file.name
    paper_id = uploaded_file.name
    logger.info(f"File uploaded: {uploaded_file.name}")
elif arxiv_id:
    source_type = "arxiv_id"
    source_identifier = arxiv_id
    paper_id = arxiv_id
    logger.info(f"ArXiv ID provided: {arxiv_id}")




if uploaded_file or arxiv_id:
    if "paper_name" not in st.session_state or st.session_state["paper_id"] != paper_id:
        logger.info(f"Processing new document. Paper ID: {paper_id}")
        # clear messages
        st.session_state.messages = []
        
        if "vector_store" in st.session_state:
            logger.info("Deleting existing vector store before processing new document")
            delete_vector_db(st.session_state["vector_store"])
            del st.session_state["vector_store"]
        
        try:
            with st.spinner("Processing Documents"):
                logger.info(f"Starting document ingestion for {source_type}: {paper_id}")
                chunks,paper_name = ingest_documents(source_type,source_identifier)
                st.session_state["paper_name"] = paper_name
                st.session_state["paper_id"] = paper_id
                logger.info(f"Creating vector database for paper: {paper_name}")
                st.session_state["vector_store"] = create_vector_db(st.session_state["paper_id"],chunks,st.session_state["embedding_model"])
                
                logger.info("Document processing completed successfully")
                st.success("processed successfully")
                 
        except Exception as e:
            logger.error(f"Error while processing document: {e}", exc_info=True)
            st.write(f"Error while processing document {e}")
            st.stop()
if "vector_store" in st.session_state:
   
    prompt = st.chat_input("Say something")
    if prompt:
        logger.info(f"User prompt received: {prompt[:100]}...")
        query= reformulate_query(st.session_state["messages"],prompt,st.session_state["model"])
        try: 
            logger.info("Retrieving relevant documents...")
            relevant_docs = retrieve_relevant_docs(query,st.session_state["vector_store"])

            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role":"user","content":prompt})

            logger.info("Generating AI response...")
            response = get_answer(prompt,st.session_state["model"],relevant_docs)
            with st.chat_message("ai"):
                st.markdown(response)
            st.session_state.messages.append({"role":"ai","content":response})
            logger.info("Chat interaction completed successfully")

        except Exception as e:
            logger.error(f"Error during chat interaction: {e}", exc_info=True)
            st.error(e)




if "paper_name" in st.session_state:
    st.sidebar.info(f"📄 Current paper: {st.session_state['paper_name']}")

