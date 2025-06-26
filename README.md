# RAGAI

This project provides a FastAPI service for uploading PDF or Word documents and storing their embeddings in a Qdrant vector database.

## Running the API
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the server:
```bash
uvicorn app.main:app --reload
```
The service uses `OLLAMA_EMBEDDING_URL`, `OLLAMA_GENERATE_URL`, `OLLAMA_MODEL`,
`OLLAMA_LLM_MODEL`, and `QDRANT_URL` environment variables. Defaults are defined in `app/main.py` and can be
overridden.

## Upload Endpoint
`POST /api/upload`

Accepts a PDF or DOCX file and stores the text segments in Qdrant. The response contains the generated `document_id` and the number of segments uploaded.

## Ask Endpoint
`POST /api/ask`

Receives a question and retrieves the top 3 relevant segments from Qdrant. The segments are used as context for Ollama to generate an answer. The response includes the answer text and referenced segments.
