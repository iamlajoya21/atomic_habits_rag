from embeddings.text_splitter import split_text
from embeddings.vector_store import store_embeddings
def main():
	print("[1/2] Splitting and processing documents...")
	chunks = split_text("data/atomic_habits.pdf")
	print("[2/2] Generating and storing embeddings...")
	store_embeddings(chunks)
	print("Embeddings stored. You can now run the Streamlit app with:\n")
	print("   streamlit run app/streamlit_app.py")
if __name__ == "__main__":
	main()