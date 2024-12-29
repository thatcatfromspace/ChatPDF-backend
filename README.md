# FastAPI Project with Ollama Integration

This repository contains a basic FastAPI application integrated with Ollama LLM for local and Dockerized deployments. Follow the steps below to set up and run the project.

---

## Features
- Upload a PDF to process and store data using a vector store.
- Ask questions and retrieve answers using Ollama's LLM.
- Local and Dockerized deployment support.

---

## Prerequisites
1. Python 3.12+
2. Docker and Docker Compose installed
3. Basic familiarity with FastAPI and Docker

---

## Project Structure
```plaintext
.
├── app/
│   ├── __init__.py
│   ├── main.py      # FastAPI entry point
│   └── ...
├── requirements.txt # Python dependencies
├── Dockerfile       # FastAPI Docker setup
├── docker-compose.yml # Docker Compose for FastAPI and Ollama
└── README.md        # Project documentation
```

---

## 1. How to Run Locally

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Install Dependencies
Ensure you have Python 3.12+ installed.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run Ollama Locally
Pull and run the Ollama Docker image:
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

Start a model (e.g., `llama3`):
```bash
docker exec -it ollama ollama run llama3
```

### Step 4: Start the FastAPI App
Run the FastAPI app:
```bash
uvicorn app.main:app --reload
```

Access the app at:
```
http://localhost:8000
```

---

## 2. How to Run with Docker

### Step 1: Build the Docker Images
```bash
docker-compose build
```

### Step 2: Start the Services
Run the containers using Docker Compose:
```bash
docker-compose up
```

- The `fastapi` service will run on `http://localhost:8000`.
- The `ollama` service will run on `http://localhost:11434`.

### Step 3: Test the Endpoints
Upload a PDF:
```bash
curl -X POST \
  http://127.0.0.1:8000/upload_pdf/ \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf"
```

Ask a question:
```bash
curl -X POST \
  http://127.0.0.1:8000/ask_question/ \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the topic?"}'
```

---

## Troubleshooting
- **Ensure Docker is Running**: Check if Docker services are running.
- **Ollama Model Loading**: Make sure the `ollama run llama3` command works and the model is properly set up.
- **Log Files**: Check logs in the `logs/` directory for detailed error information.

---

## Additional Notes
- The app uses Chroma for vector storage, persisting data in the `chroma_store` directory.
- Ollama runs on port `11434` by default. If you change the port, update the FastAPI app accordingly.

Enjoy building with FastAPI and Ollama! 🚀