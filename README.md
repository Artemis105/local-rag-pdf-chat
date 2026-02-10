# PDF Intelligence RAG Chatbot 

A Retrieval-Augmented Generation (RAG) system designed for intelligent interaction with PDF documents. While documented in English, the system's internal logic and prompts are **optimized for the Polish language**.

##  Features

* **Incremental Learning:** Automatically detects and indexes new PDF files in the `Articles/` folder without re-processing existing data.
* **Source Attribution:** Provides specific filenames and page numbers for every generated answer.
* **Contextual Memory:** Maintains a sliding window of chat history for natural follow-up questions.
* **Exportable Logs:** Saves conversations to `.txt` files with automated timestamps.

##  Tech Stack & LLM Choice

This project supports two workflows to balance privacy and performance:

| Feature | **Ollama (Default)** | **Google Gemini (Optional)** |
| :--- | :--- |:-----------------------------|
| **Execution** | 100% Local | Cloud-based                  |
| **Privacy** | Maximum (Data stays on disk) | Standard (Cloud processing)  |
| **Availability** | Unlimited (Local resources) | Limited (API Quota/Rate limits) |
| **Model** | Llama 3.2 | Gemini 2.5 Flash / Pro       |

* **Why Ollama?** It was chosen as the primary engine to avoid API rate limits and quotas encountered with Gemini's free tier. Leveraging local hardware allows for unlimited queries and ensures that sensitive PDF documents are never uploaded to external servers.
* **Framework:** LangChain using `ChromaDB` for vector storage.
* **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Optimized for Polish/English retrieval).
**Config:** * `Temperature`: Defaulted to `0.2`. This can be adjusted in `main.py` depending on the desired effect:
        * **Lower (0.0 - 0.3):** Best for factual Q&A and technical analysis (minimizes hallucinations).
        * **Higher (0.7+):** Best for creative writing or brainstorming based on the PDF content.
    * `K=5`: Retrieves the top 5 most relevant document chunks for each query.
##  Getting Started

### 1. Prerequisites
* **Ollama (Local):** [Download Ollama](https://ollama.ai/) and run `ollama run llama3.2`.
* **Gemini (Cloud):** Add your `GOOGLE_API_KEY` to the `.env` file.

### 2. Installation
Ensure you are using a virtual environment (e.g., in PyCharm), then install dependencies:
```bash
pip install -r requirements.txt
```
### 3. Project Structure

```text
├── main.py          # Chat application & RAG chain logic
├── database.py      # Vector database management (incremental updates)
├── Articles/        # Source PDF files
├── Chats/           # Exported conversation logs
├── My_base4/        # Local ChromaDB storage (ignored by Git)
└── .env             # API keys & environment variables
```