import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Importujemy nasz moduł bazy
from database import build_or_load_memory

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
def main():
    key = os.getenv("GOOGLE_API_KEY")

    if not key:
        print(" BŁĄD: Nie znaleziono klucza GOOGLE_API_KEY w pliku .env")
    else:
        #if somebody don't have ollama
        # llm = ChatGoogleGenerativeAI(
        #     model=model_name,
        #     temperature=0.1 # im niższa tym mniej zmyśla
        # )
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.2, # this can be chcanged if we want other results 0,1 facts, 0,7 creative disccussion
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

        print("\n Czat gotowy! (wpisz 'exit' by wyjść)")

        # 3. Pętla czatu
        while True:
            query = input("\n Twoje pytanie: ")
            if query.lower() in ['exit', 'quit', 'wyjdź']:
                choice= input("Czy chcesz zapisać historię rozmowydo pliku?(t/n)")
                if choice=="t":
                    if os.path.exists("./Chats"):
                        print(f"Folder Chats już istnieje")
                    else:
                        os.mkdir("./Chats")
                        print(f"Folder Chats create")

                    file_name=input("Podaj nazwę pliku ")
                    if not file_name:file_name="Chat_history"
                    try:
                        with open(f"./Chats/{file_name}.txt","w",encoding="utf-8") as f:
                            for msg in history.messages:
                                role = "TY" if msg.type=="human" else "AI"
                                f.write(f"{role}:{msg.content}\n")
                        print("Historia zapisana !")
                    except Exception as e:
                        print(f"Wystąpił błąd podczas zapisu historii :{e}")

                break
            if query.lower() == "clear":
                history = ChatMessageHistory()
                print("Historia rozmowy została wyczyszczona.")
                continue

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

                if not relevant_docs:
                    #print( "\n[INFO]: Nie znalazłem pasujących fragmentów w PDFach. Model odpowie na podstawie wiedzy ogólnej.")
                    context_text = "\n[INFO]: Nie znalazłem pasujących fragmentów w PDFach. Model odpowie na podstawie wiedzy ogólnej."
                else:
                    context_text = "\n".join([d.page_content for d in relevant_docs])


                #context_text = "\n".join([d.page_content for d in relevant_docs])

                #generate final response
                sources = {f"{os.path.basename(doc.metadata.get('source'))} [str. {doc.metadata.get('page', 0) + 1}]"
                           for doc in relevant_docs}
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

