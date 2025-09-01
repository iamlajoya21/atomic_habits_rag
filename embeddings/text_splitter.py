from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
def split_text(file_path):
	loader = PyPDFLoader(file_path)
	documents = loader.load()
	splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	return splitter.split_documents(documents)