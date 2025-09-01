from langchain_community.llms import Ollama

# This assumes you have Ollama running and have pulled the deepseek-coder:instruct model.
# If you haven't, run: `ollama pull deepseek-coder:instruct`

llm = Ollama(model="deepseek-r1:latest")
