import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

DOCS_FOLDER = "./documents"
CHROMA_PATH = "./chroma_db"

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_FOLDER):
        filepath = os.path.join(DOCS_FOLDER, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyMuPDFLoader(filepath)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                print(f"Skipping unsupported file: {filename}")
                continue
            docs.extend(loader.load())
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return docs

def ingest():
    print("Loading documents...")
    documents = load_documents()
    print(f"Total documents loaded: {len(documents)}")

    # Split into chunks (500 chars each, 50 overlap so context isn't lost)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # Create embeddings (downloads model on first run, ~90MB, then works offline)
    print("Creating embeddings (this may take a few minutes on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Save to ChromaDB
    print("Saving to vector database...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"Done! {len(chunks)} chunks saved to ChromaDB.")

if __name__ == "__main__":
    ingest()