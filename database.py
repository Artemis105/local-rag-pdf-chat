import os
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Database and input configuration
DB_PATH = "My_base4"
INPUT_PATH = "Articles"

def build_or_load_memory():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if os.path.exists(DB_PATH):
        print("Loading existing database from disk...")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


        update_database(vectorstore)
        return vectorstore

    print("Creating a new database...")
    return create_new_database(embeddings)


def create_new_database(embeddings):
    if not os.path.exists(INPUT_PATH):
        print(f"Directory {INPUT_PATH} does not exist.")
        return None

    all_splits = []

    pdf_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]

    for file in pdf_files:
        print(f"Processing: {file}")
        loader = PyPDFLoader(os.path.join(INPUT_PATH, file))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits.extend(text_splitter.split_documents(loader.load()))

    if not all_splits:
        print("No PDFs found to process.")
        return None

    # Create and persist Chroma database
    return Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=DB_PATH)


def update_database(vectorstore):
    """Update database: Add only files that are not already present."""

    existing_sources = {os.path.basename(doc['source']) for doc in vectorstore.get()['metadatas']}
    pdf_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]
    new_files = [f for f in pdf_files if f not in existing_sources]

    if not new_files:
        print("Database is up to date.")
        return

    print(f"--- Found new files: {new_files}. Updating database... ---")
    all_new_splits = []

    # Process each new file with a progress bar
    for file in tqdm(new_files, desc="Adding new files"):
        loader = PyPDFLoader(os.path.join(INPUT_PATH, file))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_new_splits.extend(text_splitter.split_documents(loader.load()))

    if all_new_splits:
        vectorstore.add_documents(all_new_splits)
        print("Database updated successfully!")