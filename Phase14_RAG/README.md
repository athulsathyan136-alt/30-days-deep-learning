# Phase14_RAG — Ask My Notes

A complete RAG (Retrieval-Augmented Generation) system that answers questions about my deep learning notes.

## How It Works

1. **Chunking** — split notes into paragraphs
2. **Embedding** — turn each chunk into a vector using `all-MiniLM-L6-v2`
3. **Retrieval** — find the 10 chunks closest to the question
4. **Re-ranking** — cross-encoder picks the best 3 chunks
5. **Generation** — LLM answers using ONLY the retrieved context

## Files

- `notes.txt` — my documents
- `rag.py` — the RAG system
- `requirements.txt` — dependencies

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-your-key-here
python rag.py