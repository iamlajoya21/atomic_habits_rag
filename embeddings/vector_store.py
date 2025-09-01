import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Define the directory where the vector store will be persisted
CHROMA_DB_PATH = './chroma_db'

def get_vector_store():
    """
    Checks if the vector store exists. If not, creates it from the PDF and
    returns the persistent Chroma vector store.
    """
    # Check if the persisted database already exists
    if os.path.exists(CHROMA_DB_PATH):
        print("Loading existing vector store from disk.")
        # Load the existing vector store with the correct embedding function
        return Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )

    # If the database does not exist, create it
    print("Creating new vector store from PDF data.")
    
    # Define the path to the PDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(os.path.dirname(current_dir), 'data', 'atomic_habits.pdf')

    # Load and split the document
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Create an Ollama embeddings instance
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create the vector store and persist it
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=ollama_embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vector_store.persist()
    print("Vector store created and persisted.")
    
    return vector_store

# if __name__ == '__main__':
#     # This will create the store the first time, and load it on subsequent runs
#     store = get_vector_store()
#     print(f"Vector store successfully loaded.")
