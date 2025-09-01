import ollama

def load_llm():
	return ollama.Chat(model="deepseek-r1")