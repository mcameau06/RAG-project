from chromadb.api.types import Document
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
#from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.document_loaders import ArxivLoader
from tempfile import NamedTemporaryFile
import streamlit as st
import os


@st.cache_resource
def load_model(model_name="gemini-2.5-flash-lite"):
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment")
    api_key = os.getenv("GOOGLE_API_KEY")
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key
           )
    return model

@st.cache_resource
def load_embedding_model(model_name="models/gemini-embedding-001"):
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment")
    api_key = os.getenv("GOOGLE_API_KEY")
    embedding_model = GoogleGenerativeAIEmbeddings(model=model_name,google_api_key=api_key)

    return embedding_model


    
def ingest_arxiv_file(arxiv_id:str):
    loader = ArxivLoader(
    query=arxiv_id,
    load_max_docs=1,
    )
    document = loader.load()
    if not document:
        raise FileNotFoundError("No documents loaded")
    return (document,document[0].metadata["Title"])

def ingest_uploaded_file(uploaded_file):

    with NamedTemporaryFile(delete=False,suffix="pdf") as temp:
                temp.write(uploaded_file.getvalue())
                pdf_path = temp.name

    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    if not document:
        raise FileNotFoundError("No documents loaded")
    return (document,uploaded_file.name)

def ingest_documents(source_type:str, source_identifier:str):
    
    if source_type == "uploaded_file":
        document,paper_name = ingest_uploaded_file(source_identifier)
        chunks = chunk_documents(document)
        return (chunks, paper_name)
    elif source_type == "arxiv_id":
        document,paper_name = ingest_arxiv_file(source_identifier)
        chunks = chunk_documents(document)
        return (chunks,paper_name)
    else:
        raise ValueError("Unknown source type")
   
def chunk_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap=200,
    add_start_index=True
    )
    
    all_splits = text_splitter.split_documents(documents)
    print(f"Split pdf into {len(all_splits)} sub-documents")

    return all_splits

def create_vector_db(collection_name,chunks, embedding_model,persist_directory="./chroma_langchain_db"):

    vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
    )
    

    document_ids = vector_store.add_documents(documents=chunks)

    print(f"Added {len(document_ids)} documents to vector store")
    print(document_ids[:3])

    return vector_store
    
def delete_vector_db(vector_store):
    try:
       
        vector_store.delete_collection()
        return True
    except Exception as e:
        print(f"Error deleting vector store: {e}")
        return False

def retrieve_relevant_docs(query, vector_store, persist_directory="./chroma_langchain_db"):
    

    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":4,"score_threshold":0.3})

    similar_docs = retriever.invoke(query)
    print("__Context__")
    for i, doc in enumerate(similar_docs):
        print(f"Document {i+1}: \n {doc.page_content}\n")
    
    return similar_docs

def reformulate_query(chat:list, query:str,model):

    if not chat:
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
        chat_history = chat_history[-10:]

    formatted_messages = contextualize_q_prompt.format_messages(chat_messages=chat_history,input=query)

    results = model.invoke(formatted_messages)

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

    combined_input = f"""based on the following documents, please answer this question: {query} 
    Documents: {chr(10).join([f"-{doc.page_content}" for doc in similar_docs])}
    Please answer like an researcher, and only answer with the information in the documents.
    If the query cannot be answered with the relevant documents, tell the user you don't know.
     """

    messages = [
        SystemMessage(content="Your a researchers assistant"),
        HumanMessage(content= combined_input)
    ]
    results = model.invoke(messages)

    print("\n __AI Response__")

    print(results.content)
    return results.content
    





