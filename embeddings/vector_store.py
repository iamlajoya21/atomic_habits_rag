from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
def store_embeddings(chunks):
	embeddings = OllamaEmbeddings(model="deepseek-r1")
	vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./vector_db")
	vector_store.persist()