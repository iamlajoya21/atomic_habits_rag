from ollama_model.deepseek_r1 import load_llm
from app.retriever import retrieve_chunks
def get_rag_response(query):
	retrieved_chunks = retrieve_chunks(query)
	context = "\n".join([chunk.page_content for chunk in retrieved_chunks])
	llm = load_llm()
	response = llm.run(f"Use the following context to answer:\n{context}\n\nQuestion: {query}")
	return response