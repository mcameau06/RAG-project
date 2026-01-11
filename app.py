from ast import List
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from numpy import histogram
import streamlit as st

load_dotenv()



@st.cache_resource
def load_model(model_name="google_genai:gemini-2.5-flash-lite"):
    model = init_chat_model(model=model_name)

    return model

@st.cache_resource
def load_embedding_model(model_name="models/gemini-embedding-001"):
    embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)

    return embedding_model

def ingest_documents(file_path:str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found {file_path}")
    # load pdf
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise FileNotFoundError("No documents loaded")
    
    return docs


def chunk_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap=200,
    add_start_index=True
    )
    
    all_splits = text_splitter.split_documents(documents)
    print(f"Split pdf into {len(all_splits)} sub-documents")

    return all_splits

def create_vector_db(chunks, embedding_model,persist_directory="./chroma_langchain_db"):

    vector_store = Chroma(
    collection_name="example_collection",
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





