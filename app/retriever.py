from langchain.vectorstores import Chroma
def retrieve_chunks(query):
	vector_store = Chroma(persist_directory="./vector_db")
	return vector_store.similarity_search(query, k=3)