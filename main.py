#pip install langchain langchain-openai openai python-dotenv
#pip install python-dotenv langchain-google-genai
#pip install pypdf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
import os


def is_file (road_to_path):
    print(" Czytam PDF i analizuję ...")

    loader = PyPDFLoader(road_to_path)
    strony = loader.load()  # To dzieli PDF na strony

    # Łączymy tekst z kilku pierwszych stron (żeby nie przeładować modelu na start)
    tekst_do_analizy = "\n".join([strona.page_content for strona in strony[:5]])

    # 3. Zadanie dla Gemini
    prompt = f"""
                     Jesteś ekspertem od sieci neuronowych. 
                     Na podstawie poniższego fragmentu:

                     {tekst_do_analizy}

                     Wykonaj następujące zadania:
                     1. Napisz krótkie streszczenie (5-8 zdań).
                     2. Wymień 3 najważniejsze pojęcia techniczne poruszone w tekście.
                     3. Przytocz najważniejsze wniosek/wnioski z tego artykułu
                      
                      Jeśli tekst jest po polsku, odpowiedz po polsku, jeśli po angielsku podaj pierw odpowiedż po angielsku a pod oddzielone polskie tłumaczenie odpowiedzi.
                     """

    try:
        odpowiedz = llm.invoke(prompt)
        print("\n--- ANALIZA PLIKU ---")
        print(odpowiedz.content)
    except Exception as e:
        print(f"Błąd: {e}")
def is_folder (road_to_path):
    print("Jestem folderem")



# 1. Ładowanie zmiennych z pliku .env
load_dotenv()
road_to_path = "Articles/2007.00047v1.pdf"
# 2. Sprawdzenie, czy klucz w ogóle się wczytał do systemu
key = os.getenv("GOOGLE_API_KEY")

if not key:
    print(" BŁĄD: Nie znaleziono klucza GOOGLE_API_KEY w pliku .env")
else:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        # 4. Krótki test
        response = llm.invoke("Odpowiedz krótko: Czy czytasz to z pliku .env?")
        print(f"Model: {response.content}")
        if os.path.isfile(road_to_path):
           is_file(road_to_path)

        elif os.path.isdir(road_to_path):
            is_folder(road_to_path)

        else:
            print(f"Nie widzę pliku {road_to_path}.")


    except Exception as e:
        print(f"Wystąpił błąd: {e}")

