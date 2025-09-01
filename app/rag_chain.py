from .retriever import retrieve_chunks
from ollama_model.deepseek_r1 import llm
from langchain_core.prompts import PromptTemplate

def get_rag_response(query: str):
    """
    Generates a RAG response by combining retrieved chunks with a prompt.
    """
    # 1. Retrieve chunks
    chunks = retrieve_chunks(query)
    chunk_text = "\n\n".join([chunk.page_content for chunk in chunks])

    # 2. Create the prompt
    prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the user's question.
    If you don't know the answer, just say that you don't know.
    
    Context:
    {context}
    
    Question: {question}
    """)
    
    formatted_prompt = prompt_template.format(context=chunk_text, question=query)

    # 3. Invoke the LLM
    response = llm.invoke(formatted_prompt)
    return response
