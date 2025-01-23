import concurrent.futures
from enum import Enum
import os
import logging
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

# Basic enum to keep track of retriever status
class Status(Enum):
    INITIALIZED = "initialized"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

FILE_STORE = "files/"
LANGUAGE_MODEL = "tinyllama"

# Logging setup
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/app.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Database setup
RETRIEVER_DB_PATH = "retrievers.db"
retriever_conn = sqlite3.connect(RETRIEVER_DB_PATH, check_same_thread=False)
retriever_cursor = retriever_conn.cursor()

RESPONSE_DB_PATH = "responses.db"
response_conn = sqlite3.connect(RESPONSE_DB_PATH, check_same_thread=False)
response_cursor = response_conn.cursor()

retriever_cursor.execute(f"""
CREATE TABLE IF NOT EXISTS retrievers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    chroma_dir TEXT NOT NULL,
    status TEXT DEFAULT {Status.INITIALIZED.value} NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    UNIQUE(user_id, file_path) ON CONFLICT REPLACE   
)
""")
retriever_conn.commit()

response_cursor.execute(f"""
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT DEFAULT {Status.INITIALIZED.value} NOT NULL,
    response TEXT DEFAULT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    UNIQUE(user_id, file_path) ON CONFLICT REPLACE
)
""")
response_conn.commit()

# App setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-planet-chatbot.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prompt setup
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""You are a knowledgeable chatbot. Provide helpful answers.
    Context: {context}
    History: {history}
    User: {question}
    Chatbot:""",
)

# Maintain separate memories for each user and each file
memory_map = {}
def get_or_create_memory(user_id: str, file_path: str):
    key = (user_id, file_path)
    if key not in memory_map:
        memory_map[key] = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question"
        )
    return memory_map[key]

# LLM and embeddings setup
llm = OllamaLLM(
    base_url="http://localhost:11434/",
    model=LANGUAGE_MODEL,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
embedding_function = OllamaEmbeddings(base_url="http://localhost:11434/", model=LANGUAGE_MODEL) # NOT a good model enough model - used only because of self hosted LLM

# Clear memory if needed - do not mix previously requested documents with new ones
def clear_conversation_memory(user_id: str, file_path: str):
    key = (user_id, file_path)
    if key in memory_map:
        memory_map[key].clear()

# Save and retrieve retriever from database
def save_retriever(user_id: str, file_path: str, chroma_dir: str):
    retriever_cursor.execute(
        "INSERT INTO retrievers (user_id, file_path, chroma_dir) VALUES (?, ?, ?)",
        (user_id, file_path, chroma_dir),
    )
    retriever_conn.commit()

def get_retriever(user_id: str, file_path: str):
    retriever_cursor.execute("SELECT chroma_dir, status FROM retrievers WHERE user_id = ? AND file_path = ?", (user_id, file_path,))
    result = retriever_cursor.fetchone()
    if result and result[1] == Status.COMPLETED.value:
        chroma_dir = result[0]
        return Chroma(persist_directory=chroma_dir, embedding_function=embedding_function).as_retriever()
    return None

def update_retriever_status(user_id: str, file_path: str, status: Status):
    retriever_cursor.execute("UPDATE retrievers SET status = ? WHERE user_id = ? AND file_path = ?", (status.value, user_id, file_path))
    retriever_conn.commit()

def check_file_exists(file_path: str):
    return os.path.exists(file_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF QA API!"}

# Helper function 
def process_chunking(document_part):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200) # chunk_size should be around 1500 characters and there must be overlap of ~200
                                                                                       # but current parameters have been applied to optimize for hardware limitations
    all_splits = text_splitter.split_documents(document_part)
    return all_splits

# Parallelizing the chunking process. Chunking is parallelized to improve performance.
# We split the document into 4 parts and processes each part in parallel 
# using ThreadPoolExecutor and store the results into the vector database.
def parallel_chunking(file_path):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    data = loader.load()

    num_chunks = len(data)  
    num_parts = 4 
    chunk_size = max(1, num_chunks // num_parts) # max() used because fails if num_chunks < num_parts

    document_parts = [data[i:i + chunk_size] for i in range(0, num_chunks, chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parts) as executor: # using threads over processes because chunking is I/O bound
        results = list(executor.map(process_chunking, document_parts))

    parallel_splits = [chunk for result in results for chunk in result]
    return parallel_splits

# Move PDF chunking to background - giving immediate upload response times while easing the load on the server
def background_chunker(file_path, chroma_dir, user_id):
    try:
        parallel_splits = parallel_chunking(file_path) 
        Chroma.from_documents(
            documents=parallel_splits,
            embedding=embedding_function,
            persist_directory=chroma_dir,
        )

        print("File processing completed.")
        update_retriever_status(user_id, file_path, Status.COMPLETED)
        logging.info(f"Retriever successfully initialized for user: {user_id}")
    except Exception as e:
        logging.error(f"Error processing PDF in background: {str(e)}")

def save_response(user_id, file_path, response):
    response_cursor.execute(
        "INSERT INTO responses (user_id, file_path, response) VALUES (?, ?, ?)",
        (user_id, file_path, response),
    )
    response_conn.commit()

def get_response(user_id, file_path):
    response_cursor.execute("SELECT response FROM responses WHERE user_id = ? AND file_path = ?", (user_id, file_path,))
    result = response_cursor.fetchone()
    if result:
        return result[0]
    return None

def get_response_from_model(user_id, file_path, question):
    retriever = get_retriever(user_id, file_path)
    if not retriever:
        return None
    # Use or create a memory for each user-file pair
    user_file_memory = get_or_create_memory(user_id, file_path)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        chain_type_kwargs={"prompt": prompt, "memory": user_file_memory},
    )
    response = qa_chain.invoke(question)
    return response["result"]

def background_response_processor(user_id, file_path, question):
    try:
        question = get_response_from_model(user_id, file_path, question)
        save_response(user_id, file_path, question)
        logging.info(f"Response saved for user: {user_id}")
    except Exception as e:
        logging.error(f"Error saving response in background: {str(e)}")

def update_response_status(user_id: str, file_path: str, status: str):
    response_cursor.execute("UPDATE responses SET status = ? WHERE user_id = ? AND file_path = ?", (status, user_id, file_path))
    response_conn.commit()

@app.post("/api/clear_memory/")
async def clear_memory_endpoint(user_id: str, file_path: str):
    clear_conversation_memory(user_id, file_path)
    return {"message": "Memory cleared."}

@app.get("/api/upload_status/")
async def upload_status(file_path: str, user_id: str = "default_user"):
    retriever_cursor.execute("SELECT status FROM retrievers WHERE user_id = ? AND file_path = ?", (user_id, f"{FILE_STORE}{file_path}"))
    result = retriever_cursor.fetchone()
    if not result:
        return {"status": "No PDF uploaded."}
    return {"file": file_path.lstrip(FILE_STORE), "status": result[0]}

@app.get("/api/get_files/")
async def get_files(user_id: str = "default_user"):
    retriever_cursor.execute("SELECT file_path FROM retrievers WHERE user_id = ?", (user_id,))
    result = retriever_cursor.fetchall()
    if not result:
        return {"files": []}
    return {"files": list(file[0].lstrip(FILE_STORE) for file in result)}

# File upload endpoint
@app.post("/api/upload_pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), user_id: str = "default_user"):
    try:
        file_path = f"{FILE_STORE}{file.filename}"
        if check_file_exists(file_path):
            return {"message": "PDF already uploaded."}
        
        chroma_dir = f"chroma_store/{user_id}/{file.filename[0:16]}"

        os.makedirs('files', exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        save_retriever(user_id, file_path, chroma_dir)
        background_tasks.add_task(background_chunker, file_path, chroma_dir, user_id)
        update_retriever_status(user_id, file_path, Status.PROCESSING)
        return {"message": "PDF uploaded and processing started."}
    
    except Exception as e:
        logging.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

# Question-answering endpoint
class QuestionRequest(BaseModel):
    question: str

@app.post("/api/ask_question/")
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks, file: str, user_id: str = "default_user"):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="PDF file not provided.")
        retriever = get_retriever(user_id, f"{FILE_STORE}{file}")
        if not retriever:
            raise HTTPException(status_code=400, detail="PDF not uploaded.")
        background_tasks.add_task(background_response_processor, user_id, f"{FILE_STORE}{file}", request.question)
        return {"message": "Question received. Processing response in the background."}
    except Exception as e:
        logging.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.get("/api/response_status/")
async def response_status(file_path: str, user_id: str = "default_user"):
    response_cursor.execute("SELECT status FROM responses WHERE user_id = ? AND file_path = ?", (user_id, f"{FILE_STORE}{file_path}"))
    result = response_cursor.fetchone()
    if not result:
        return {"status": "No record of conversation."}
    if result[0] == Status.PROCESSING.value:
        return {"status": result[0]}
    if result[0] == Status.COMPLETED.value:
        response = get_response(user_id, f"{FILE_STORE}{file_path}")
        return {"status": result[0], "response": response}
    return {"status": result[0]}

# Global error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, e: HTTPException):
    logging.error(f"HTTP Exception: {e.detail}")
    return JSONResponse(
        status_code=e.status_code, content={"message": e.detail}
    )