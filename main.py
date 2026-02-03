#pip install langchain langchain-openai openai python-dotenv
#pip install python-dotenv langchain-google-genai
#pip install pypdf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
import os
import csv


def check_csv(csv_path):
    """Zwraca listę (set) plików, które już są w CSV."""
    if not os.path.exists(csv_path):
        return set()
    processed = set()
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)
        for row in reader:
            if row and len(row)>0:
                file_name=row[0].strip().replace('""','')
                processed.add(file_name)
    return processed
def get_pdf_text(path):
    loader = PyPDFLoader(path)
    pages = loader.load()

    return "\n".join([p.page_content for p in pages])

def is_file (road_to_path):

    tekst_do_analizy=get_pdf_text(road_to_path)

    prompt = f"""
        Jesteś rzetelnym asystentem naukowym. 
        Na podstawie poniższego tekstu:
        ---
        {tekst_do_analizy}
        ---
        Wykonaj zadania:
        1. Cel artykułu (3-5 zdań).
        2. 3 najważniejsze pojęcia techniczne (nie tłumacz ich nazw).
        3. Najważniejsze wnioski.

        Zasada języka:
        - Jeśli tekst jest po POLSKU: odpowiedz tylko po polsku.
        - Jeśli tekst jest po ANGIELSKU: podaj odpowiedź najpierw po angielsku, a poniżej jej pełne polskie tłumaczenie.
        """

    try:
        odpowiedz = llm.invoke(prompt)
        print("\n--- ANALIZA PLIKU ---")
        print(odpowiedz.content)
    except Exception as e:
        print(f"Błąd: {e}")


def gemini_to_csv(path, llm, plik):
    text=get_pdf_text(os.path.join(path, plik))
    prompt = f"""
    Jesteś asystentem naukowym. Przeanalizuj tekst i zwróć dane w formacie:
    TYTUŁ|AUTOR|CEL|WNIOSKI

    Zasady:
    - Jeśli tekst jest po angielsku, CEL i WNIOSKI napisz po polsku.
    - TYTUŁ i AUTOR zostaw w oryginale.
    - Użyj DOKŁADNIE znaku | jako separatora (tylko 3 znaki | w całej odpowiedzi).

    Tekst: {text[:30000]}  # Ograniczenie do ok. 30k znaków dla stabilności
    """

    response = llm.invoke(prompt)
    raw_text=response.content
    try:

        parts = raw_text.split('|')
    # Usuwamy ewentualne zbędne napisy typu "TYTUŁ:"
        clean_parts = [p.replace("TYTUŁ:", "").replace("AUTOR:", "").replace("CEL:", "").replace("WNIOSKI:", "").strip()
                   for p in parts]
        if len(clean_parts)<4:
            clean_parts+= [""]*(4-len(clean_parts))
        return [plik] + clean_parts[:4]

    except Exception as e:
         return [plik, "Błąd formatowania", "", "", str(e)]



load_dotenv()
plik_wynikowy = "podsumowanie_bibliografii.csv"
road_to_path = "Articles"# /2007.00047v1.pdf"
already_check= check_csv(plik_wynikowy)
# Sprawdzenie, czy klucz w ogóle się wczytał do systemu
key = os.getenv("GOOGLE_API_KEY")

if not key:
    print(" BŁĄD: Nie znaleziono klucza GOOGLE_API_KEY w pliku .env")
else:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite"
        )
        if os.path.isfile(road_to_path):
           is_file(road_to_path)

        elif os.path.isdir(road_to_path):
            plik_istnieje = os.path.isfile(plik_wynikowy)

            pliki = [f for f in os.listdir(road_to_path) if f.lower().endswith('pdf')]

            with open(plik_wynikowy, mode ='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')

                # Zapisujemy nagłówek TYLKO jeśli plik jest nowy
                if not plik_istnieje:
                    writer.writerow(['Plik', 'Tytuł', 'Autor', 'Cel badania', 'Główne wnioski'])

                for plik in pliki:
                    if plik in already_check:
                        print(f"Pomijam zapisany plik: {plik} ")
                        continue
                    print(f"Pracuję nad plikiem :{plik}...")
                    row=gemini_to_csv(road_to_path,llm,plik)
                    writer.writerow(row)

        else:
            print(f"Nie znalazłem pliku")


    except Exception as e:
        print(f"Wystąpił błąd: {e}")

