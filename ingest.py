import os
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredODTLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

os.environ["ANONYMIZED_TELEMETRY"] = "False"

DOCS_FOLDER = "./documents"
CHROMA_PATH = "./chroma_db"

LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".odt": UnstructuredODTLoader,
}

def load_documents():
    docs = []
    skipped = []
    for filename in sorted(os.listdir(DOCS_FOLDER)):
        filepath = os.path.join(DOCS_FOLDER, filename)
        ext = os.path.splitext(filename)[1].lower()

        loader_cls = LOADERS.get(ext)
        if not loader_cls:
            skipped.append(filename)
            continue

        try:
            # TextLoader needs encoding arg, others don't
            if ext == ".txt":
                loader = loader_cls(filepath, encoding="utf-8")
            else:
                loader = loader_cls(filepath)
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  ✓ {filename} ({len(loaded)} pages/sections)")
        except Exception as e:
            print(f"  ✗ {filename} — ERROR: {e}")

    if skipped:
        print(f"\nSkipped (unsupported): {', '.join(skipped)}")
    return docs

def ingest():
    print("Loading documents from ./documents ...")
    documents = load_documents()
    print(f"\nTotal documents loaded: {len(documents)}")

    if not documents:
        print("No documents to ingest. Add files to ./documents and try again.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("Loading embedding model (downloads ~90 MB on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building vector database...")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"\n✓ Done. {len(chunks)} chunks saved to {CHROMA_PATH}")

if __name__ == "__main__":
    ingest()