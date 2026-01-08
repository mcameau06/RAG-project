import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import tool
from langchain.agents import create_agent
load_dotenv()


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
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space":"cosine"}
    )
    document_ids = vector_store.add_documents(documents=chunks)

    print(f"Added {len(document_ids)} documents to vector store")
    print(document_ids[:3])

    return vector_store


def retrieve_relevant_docs(embedding_model,query, persist_directory="./chroma_langchain_db"):
    
    db = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":4,"score_threshold":0.3})

    similar_docs = retriever.invoke(query)
    print("__Context__")
    for i, doc in enumerate(similar_docs):
        print(f"Document {i+1}: \n {doc.page_content}\n")
    
    return similar_docs
        
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
        

def main():
   embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
   model = init_chat_model("google_genai:gemini-2.5-flash-lite")
   documents =  ingest_documents("paper.pdf")
   chunks  = chunk_documents(documents)

   vectorstore = create_vector_db(chunks,embedding_model)
   query = "How do you create the transformer? Explain to me like I'm a highschooler"
   relevant_docs = retrieve_relevant_docs(embedding_model,query)
   query_2 = "Can dogs fly?"

   get_answer(query,model,relevant_docs)
   get_answer(query_2,model,relevant_docs)
if __name__ == "__main__":
    main()




