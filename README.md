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
Create a `.env` file to configure the service:
```
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama2
```
These values are loaded via `python-dotenv` and override the defaults defined in
`app/main.py`.

## Upload Endpoint
`POST /api/upload`

Accepts a PDF or DOCX file and stores the text segments in Qdrant. Each chunk is stored with `chunk_index`, `file_name` and `upload_time` metadata. The response contains the generated `document_id` and the number of segments uploaded.

## Ask Endpoint
`POST /api/ask`

Receives a question and retrieves the top 5 relevant segments from Qdrant. A `rerank` function sorts them and the top 3 are used as context for Ollama to generate an answer. The response includes the answer text and referenced segments.
