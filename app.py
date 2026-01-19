from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.document_loaders import ArxivLoader
from tempfile import NamedTemporaryFile
import streamlit as st
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()
@st.cache_resource
def load_model(model_name="gemini-2.5-flash-lite"):
    logger.info(f"Loading model: {model_name}")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment")
        raise RuntimeError("GOOGLE_API_KEY not found in environment")
    
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key
           )
    logger.info(f"Model {model_name} loaded successfully")
    return model

@st.cache_resource
def load_embedding_model(model_name="models/gemini-embedding-001"):
    logger.info(f"Loading embedding model: {model_name}")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment")
        raise RuntimeError("GOOGLE_API_KEY not found in environment")
    embedding_model = GoogleGenerativeAIEmbeddings(model=model_name,google_api_key=api_key)
    logger.info(f"Embedding model {model_name} loaded successfully")
    return embedding_model


    
def ingest_arxiv_file(arxiv_id:str):
    logger.info(f"Ingesting ArXiv document with ID: {arxiv_id}")
    loader = ArxivLoader(
    query=arxiv_id,
    load_max_docs=1,
    )
    document = loader.load()
    if not document:
        logger.error(f"No documents loaded for ArXiv ID: {arxiv_id}")
        raise FileNotFoundError("No documents loaded")
    paper_title = document[0].metadata["Title"]
    logger.info(f"Successfully loaded ArXiv document: {paper_title}")
    return (document, paper_title)

def ingest_uploaded_file(uploaded_file):
    logger.info(f"Ingesting uploaded file: {uploaded_file.name}")
    with NamedTemporaryFile(delete=False,suffix="pdf") as temp:
                temp.write(uploaded_file.getvalue())
                pdf_path = temp.name
    logger.debug(f"Temporary file created at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    if not document:
        logger.error(f"No documents loaded from file: {uploaded_file.name}")
        raise FileNotFoundError("No documents loaded")
    logger.info(f"Successfully loaded {len(document)} pages from {uploaded_file.name}")
    return (document,uploaded_file.name)

def ingest_documents(source_type:str, source_identifier:str):
    logger.info(f"Ingesting documents from source type: {source_type}")
    if source_type == "uploaded_file":
        document,paper_name = ingest_uploaded_file(source_identifier)
        chunks = chunk_documents(document)
        logger.info(f"Document ingestion complete. Generated {len(chunks)} chunks")
        return (chunks, paper_name)
    elif source_type == "arxiv_id":
        document,paper_name = ingest_arxiv_file(source_identifier)
        chunks = chunk_documents(document)
        logger.info(f"Document ingestion complete. Generated {len(chunks)} chunks")
        return (chunks,paper_name)
    else:
        logger.error(f"Unknown source type: {source_type}")
        raise ValueError("Unknown source type")
   
def chunk_documents(documents):
    logger.info(f"Chunking {len(documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap=200,
    add_start_index=True
    )
    
    all_splits = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(all_splits)} sub-documents")

    return all_splits





def create_vector_db(collection_name,chunks, embedding_model):

    sanitized_collection_name = sanitize_collection_name(collection_name)
    logger.info(f"Using sanitized collection name: {sanitized_collection_name}")

    vector_store = Chroma(
    collection_name=sanitized_collection_name,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
    )
    logger.info(f"Chroma vector store created for collection: {sanitized_collection_name}")

    logger.info(f"Adding {len(chunks)} chunks to vector store...")
    document_ids = vector_store.add_documents(documents=chunks)

    logger.info(f"Successfully added {len(document_ids)} documents to vector store")
    logger.debug(f"First 3 document IDs: {document_ids[:3]}")

    return vector_store


def sanitize_collection_name(name:str) -> str:
    """ Ensures that the collection name is alphanumeric"""
    original_name = name
    logger.debug(f"Sanitizing collection name: {original_name}")
    
    name = name.replace(" ", "_")
    name = name.strip('._-')

    if len(name) < 3:
        name = f"doc_{name}" if name else "collection"
        logger.debug(f"Collection name too short, adjusted to: {name}")
    
    if len(name) > 512:
        name = name[:512]
        name = name.strip('._-')
        logger.debug(f"Collection name too long, truncated to: {name}")

    if original_name != name:
        logger.info(f"Collection name sanitized: '{original_name}' -> '{name}'")
    return name


def delete_vector_db(vector_store):
    try:
        logger.info("Deleting vector store collection...")
        vector_store.delete_collection()
        logger.info("Vector store collection deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting vector store: {e}", exc_info=True)
        return False

def retrieve_relevant_docs(query, vector_store, persist_directory="./chroma_langchain_db"):
    logger.info(f"Retrieving relevant documents for query: {query[:100]}...")
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":4,"score_threshold":0.3})

    similar_docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(similar_docs)} relevant documents")
    logger.debug("Retrieved document context:")
    for i, doc in enumerate(similar_docs):
        logger.debug(f"Document {i+1}: {doc.page_content[:200]}...")
    
    return similar_docs

def reformulate_query(chat:list, query:str,model):
    logger.debug(f"Reformulating query. Chat history length: {len(chat)}")
    if not chat:
        logger.debug("No chat history, returning original query")
        return query
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_messages"),
        ("human", "{input}"),
    ]
)   

    chat_history = get_conversation_history(chat)
    if len(chat_history) > 10:
        logger.debug(f"Chat history too long ({len(chat_history)}), truncating to last 10 messages")
        chat_history = chat_history[-10:]

    formatted_messages = contextualize_q_prompt.format_messages(chat_messages=chat_history,input=query)

    logger.debug("Invoking model to reformulate query...")
    results = model.invoke(formatted_messages)
    logger.debug(f"Reformulated query: {results.content[:100]}...")

    return results.content
    

def get_conversation_history(chat:list) -> list[HumanMessage | AIMessage]:
    conversation_history = []

    for message in chat:
        if  message["role"]== "user":
            conversation_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "ai":
            conversation_history.append(AIMessage(content=message["content"]))

    return conversation_history

def get_answer(query, model, similar_docs):
    logger.info(f"Generating answer for query: {query[:100]}...")
    logger.debug(f"Using {len(similar_docs)} relevant documents as context")
    
    combined_input = f"""based on the following documents, please answer this question: {query} 
    Documents: {chr(10).join([f"-{doc.page_content}" for doc in similar_docs])}
    Please answer like an researcher, and only answer with the information in the documents.
    If the query cannot be answered with the relevant documents, tell the user you don't know.
     """

    messages = [
        SystemMessage(content="Your a researchers assistant"),
        HumanMessage(content= combined_input)
    ]
    logger.debug("Invoking model to generate answer...")
    results = model.invoke(messages)

    logger.info("AI response generated successfully")
    logger.debug(f"AI Response: {results.content[:200]}...")
    return results.content
    





