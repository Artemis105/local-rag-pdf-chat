import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- KONFIGURACJA ---
load_dotenv()
DB_PATH = "My_base3"
INPUT_PATH= "Articles"
model_name= "gemini-2.0-flash-lite"#"gemini-2.5-flash"

def build_or_load_memory():
    # Model lokalny - pobierze się raz (ok. 100MB) i działa  na dysku bez koniecznosci wysyłania do chmury
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        all_splits.extend(text_splitter.split_documents(docs))

    if not all_splits:
        print(" Nie znaleziono PDFów do przetworzenia.")
        return None

    # Tworzenie bazy
    vectorstore = Chroma.from_documents(documents=all_splits,embedding=embeddings,persist_directory=DB_PATH)
    print("Baza gotowa!")
    return vectorstore


def main():
    key = os.getenv("GOOGLE_API_KEY")

    if not key:
        print(" BŁĄD: Nie znaleziono klucza GOOGLE_API_KEY w pliku .env")
    else:

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1 # im niższa tym mniej zmyśla
        )
        vectorstore = build_or_load_memory()
        history = ChatMessageHistory()
        # prompt= """Jesteś pomocnym asystentem. Odpowiadaj po polsku.
        #     Traktuj HISTORIĘ ROZMOWY tylko jako dodatkowy kontekst.
        #     Jeśli nowe pytanie dotyczy zupełnie innego tematu niż historia, zignoruj historię i skup się wyłącznie na KONTEKŚCIE Z PDF.
        #     Jeśli w tekście nie ma odpowiedzi, napisz rzetelnie, że nie posiadasz takich informacji.
        #
        #
        #     HISTORIA ROZMOWY:{chat_history}
        #     Kontekst:{context}
        #     Pytanie: {question}
        #   """
        prompt = """Jesteś ekspertem AI. Odpowiadaj zawsze po polsku, krótko i konkretnie.
        Korzystaj z KONTEKSTU, aby odpowiedzieć na pytanie. Jeśli w KONTEKŚCIE nie ma odpowiedzi, użyj własnej wiedzy, ale zaznacz to.

        KONTEKST: {context}
        HISTORIA: {chat_history}

        Pytanie: {question}
        """


        prompt2=ChatPromptTemplate.from_template(prompt)
        retriever=vectorstore.as_retriever(search_kwargs={"k":5})


        # 2. Tworzymy narzędzie do zadawania pytań
        rag_chain = (
                {
                    # "Weź artykuły"
                    "context": lambda x: x["context"],
                    # get original question"
                    "question": lambda x: x["question"],

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
                chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages[-4:]])
                #we must change a question into question which computer can uderstand
                if chat_history: # if history is empty copy question, else generate new question
                    context_prompt = f"""Na podstawie historii i pytania, stwórz pytanie do wyszukiwarki. 
                    Bądź precyzyjny. Jeśli użytkownik prosi o rozwinięcie, określ dokładnie jaki temat z historii ma zostać rozwinięty.
                    Historia: {chat_history}
                    Pytanie: {query}"""

                    search_query = llm.invoke(context_prompt).content
                else:
                    search_query=query

                relevant_docs = retriever.invoke(search_query)
                context_text = "\n".join([d.page_content for d in relevant_docs])

                #generate final response
                sources = {doc.metadata.get('source', 'nieznane') for doc in relevant_docs}

                response = rag_chain.invoke({
                    "question": query,
                    "chat_history": chat_history,
                    "context":  context_text # dokumenty z search_query
                })

                history.add_user_message(query)
                history.add_ai_message(response)
                print(f"\n ODPOWIEDŹ: {response}")
                print(f"Źródła: {', '.join(sources)}")
            except Exception as e:
                print(f"Wystąpił błąd podczas rozmowy: {e}")






if __name__ == "__main__":
    main()

