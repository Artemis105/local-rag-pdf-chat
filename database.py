import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

DB_PATH = "My_base4"
INPUT_PATH = "Articles"

def build_or_load_memory():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if os.path.exists(DB_PATH):
        print("Ładuję gotową bazę z dysku")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

        update_database(vectorstore) #douczanie
        return vectorstore

    print("Tworzę nową bazę")
    return create_new_database(embeddings)


def create_new_database(embeddings):
    if not os.path.exists(INPUT_PATH):
        print(f"Podany folder nie istnieje")
        return None

    all_splits = []
    pdf_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]

    for file in pdf_files:
        print(f"Przetwarzam: {file}")
        loader = PyPDFLoader(os.path.join(INPUT_PATH, file))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits.extend(text_splitter.split_documents(loader.load()))

    if not all_splits:
        print(" Nie znaleziono PDFów do przetworzenia.")
        return None

    return Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=DB_PATH)


def update_database(vectorstore):
    """Douczanie: Dodaje tylko pliki, których jeszcze nie ma w bazie."""

    existing_sources = {os.path.basename(doc['source']) for doc in vectorstore.get()['metadatas']}

    pdf_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]

    new_files = [f for f in pdf_files if f not in existing_sources]

    if not new_files:
        print("Baza jest aktualna.")
        return

    print(f"--- Znaleziono nowe pliki: {new_files}. Douczam bazę... ---")
    all_new_splits = []
    for file in tqdm(new_files, desc="Dodawanie nowych plików"):
        loader = PyPDFLoader(os.path.join(INPUT_PATH, file))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_new_splits.extend(text_splitter.split_documents(loader.load()))

    if all_new_splits:
        vectorstore.add_documents(all_new_splits)
        print("Baza została zaktualizowana!")

