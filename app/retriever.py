from embeddings.vector_store import get_vector_store

def retrieve_chunks(query: str):
    """
    Retrieves relevant document chunks from the vector store based on a query.
    """
    vector_store = get_vector_store()
    # Perform a similarity search to get the most relevant chunks
    retrieved_chunks = vector_store.similarity_search(query, k=3)
    return retrieved_chunks
