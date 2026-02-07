import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
# --- KONFIGURACJA ---
load_dotenv()
DB_PATH = "My_base"
INPUT_PATH= "Articles"
model_name= "gemma-3-1b-it"

def build_or_load_memory():
    # Model lokalny - pobierze się raz (ok. 100MB) i działa  na dysku bez koniecznosci wysyłania do chmury
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_PATH):
        print(" Ładuję gotową bazę z dysku...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    print(" Tworzę nową bazę lokalnie (bez limitów Google!)...")
    if not os.path.exists(INPUT_PATH):
        print(f" Folder {INPUT_PATH} nie istnieje!")
        return None

    all_splits = []
    pdf_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]

    for file in pdf_files:
        print(f"Czytam: {file}")
        loader = PyPDFLoader(os.path.join(INPUT_PATH, file))
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits.extend(text_splitter.split_documents(docs))

    if not all_splits:
        print(" Nie znaleziono PDFów do przetworzenia.")
        return None

    # Tworzenie bazy
    vectorstore = Chroma.from_documents(documents=all_splits,embedding=embeddings,persist_directory=DB_PATH)
    print("✅ Baza gotowa!")
    return vectorstore
def main():
    key = os.getenv("GOOGLE_API_KEY")

    if not key:
        print(" BŁĄD: Nie znaleziono klucza GOOGLE_API_KEY w pliku .env")
    else:

        llm = ChatGoogleGenerativeAI(
            model=model_name
        )
        vectorstore = build_or_load_memory()
        history = ChatMessageHistory()
        # prompt= """Jesteś pomocnym asystentem. Odpowiadaj po polsku.
        #     Traktuj HISTORIĘ ROZMOWY tylko jako dodatkowy kontekst.
        #     Jeśli nowe pytanie dotyczy zupełnie innego tematu niż historia, zignoruj historię i skup się wyłącznie na KONTEKŚCIE Z PDF.
        #     Jeśli w tekście nie ma odpowiedzi, napisz rzetelnie, że nie posiadasz takich informacji.
        prompt = """Jesteś precyzyjnym asystentem naukowym. 
        Twoim głównym zadaniem jest odpowiadanie na podstawie dostarczonego KONTEKSTU Z PDF.

        Zasady:
        1. Priorytetem jest KONTEKST Z PDF. 
        2. HISTORIĘ ROZMOWY traktuj tylko jako pomoc, by zrozumieć do czego odnoszą się zaimki (np. "to", "on", "poprzedni").
        3. Jeśli użytkownik zmienia temat i zadaje pytanie niezwiązane z historią, zignoruj historię i odpowiedz tylko na podstawie PDF.
        4. Jeśli w PDF nie ma odpowiedzi, napisz: "Nie znajduję informacji na ten temat w moich dokumentach".
        
            HISTORIA ROZMOWY:{chat_history}
            Kontekst:{context}
            Pytanie: {question}
          """
        prompt2=ChatPromptTemplate.from_template(prompt)
        retriever=vectorstore.as_retriever(search_kwargs={"k":5})


        # 2. Tworzymy narzędzie do zadawania pytań
        rag_chain = (
                {
                    # Lambda mówi: "Weź pytanie z wejścia i użyj go, by wyciągnąć tekst z PDF"
                    "context": lambda x: retriever.invoke(x["question"]),
                    # Lambda mówi: "Po prostu przekaż pytanie dalej do promptu"
                    "question": lambda x: x["question"],
                    # Lambda mówi: "Po prostu przekaż sformatowaną historię dalej"
                    "chat_history": lambda x: x["chat_history"]
                }
                | prompt2
                | llm
                | StrOutputParser()
        )

        print("\n Cześć! Przeczytałem Twoje artykuły. O co chcesz zapytać? (wpisz 'exit' by wyjść)")

        # 3. Pętla czatu
        while True:
            query = input("\n Twoje pytanie: ")
            if query.lower() in ['exit', 'quit', 'wyjdź']:
                break
            try:
                relevant_docs = retriever.invoke(query)
                sources = {doc.metadata.get('source', 'nieznane') for doc in relevant_docs}
                chat_history= "\n".join([F"{msg.type}: {msg.content}" for msg in history.messages[-6:]])

                response = rag_chain.invoke({
                    "question": query,
                    "chat_history": chat_history
                })

                history.add_user_message(query)
                history.add_ai_message(response)
                print(f"\n ODPOWIEDŹ: {response}")
                print(f"Źródła: {', '.join(sources)}")
            except Exception as e:
                print(f"Wystąpił błąd podczas rozmowy: {e}")






if __name__ == "__main__":
    main()

