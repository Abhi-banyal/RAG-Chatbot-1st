# RAG Chatbot

A document-based RAG chatbot built with FastAPI, LangChain, FAISS, and a React + Vite frontend.

It supports:
- PDF, TXT, PNG, JPG, and JPEG uploads
- Automatic indexing when a file is uploaded
- Question answering over all indexed documents
- Optional multimodal handling for images and PDF pages with visual content

## Project Structure

- `app/` - FastAPI backend, ingestion pipeline, retriever, and chatbot logic
- `frontend/` - React UI built with Vite
- `data/` - Sample source documents that ship with the project
- `uploads/` - New files uploaded from the UI
- `vectorstore/` - Persisted FAISS index

## Requirements

- Python 3.11+
- Node.js 18+
- An Azure OpenAI deployment for chat responses

## Environment Variables

Create a `.env` file in the project root with:

```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-xx-xx
AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment-name

# Optional, for frontend access from different origins
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## Install

### Backend

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### Frontend

```powershell
cd frontend
npm install
```

## Run The App

### Start Backend

From the project root:

```powershell
uvicorn app.main:app --reload
```

Backend defaults to:
- API: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

### Start Frontend

From `frontend/`:

```powershell
npm run dev
```

Frontend defaults to:
- `http://localhost:5173`

## How Uploading Works

The upload flow is optimized to avoid reprocessing the full document set.

When a user uploads a file:
1. The file is saved into `uploads/`
2. Only that new file is extracted
3. Only that file is chunked
4. Only the new chunks are embedded
5. The new chunks are appended to the existing FAISS vectorstore

This means old indexed files are not deleted or reprocessed on every upload.

## API Endpoints

- `GET /health` - backend health check
- `GET /documents` - list indexed files
- `POST /upload` - upload and index new files
- `POST /chat` - ask a question
- `POST /ingest` - manual full rebuild of the vectorstore
- `GET /docs` - Swagger UI
- `GET /openapi.json` - OpenAPI spec

## Ingestion Notes

- Text files are read directly
- PDFs are extracted page by page
- PNG/JPG/JPEG files are converted into retrieval documents using the existing image-processing path
- Metadata is preserved for source filename, page number, figure number, chunk ID, and file type

## Testing

Run the test suite from the project root:

```powershell
python -m unittest discover tests
```

## Useful Commands

### Rebuild the full vectorstore manually

```powershell
python -m app.ingest
```

## Notes

- If the FAISS vectorstore does not exist yet, upload a file or run the full ingest once.
- If Azure OpenAI settings are missing, chat responses will fall back to the project's fallback behavior.
