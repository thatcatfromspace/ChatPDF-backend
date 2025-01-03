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
LLM_MODEL = "mistral"

# Logging setup
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/app.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Database setup
db_path = "retrievers.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute(f"""
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
conn.commit()

# App setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-planet-chatbot.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prompt and memory setup
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""You are a knowledgeable chatbot. Provide helpful answers.
    Context: {context}
    History: {history}
    User: {question}
    Chatbot:""",
)
memory = ConversationBufferMemory(
    memory_key="history", return_messages=True, input_key="question"
)

# LLM and embeddings setup
llm = OllamaLLM(
    base_url="http://localhost:11434/",
    model=LLM_MODEL,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
embedding_function = OllamaEmbeddings(base_url="http://localhost:11434/", model=LLM_MODEL) # NOT a good model enough model - used only because of self hosted LLM


# Save and retrieve retriever from database
def save_retriever_to_db(user_id: str, file_path: str, chroma_dir: str):
    cursor.execute(
        "INSERT INTO retrievers (user_id, file_path, chroma_dir) VALUES (?, ?, ?)",
        (user_id, file_path, chroma_dir),
    )
    conn.commit()

def get_retriever_from_db(user_id: str, file_path: str):
    cursor.execute("SELECT chroma_dir, status FROM retrievers WHERE user_id = ? AND file_path = ?", (user_id, file_path,))
    result = cursor.fetchone()
    if result and result[1] == Status.COMPLETED.value:
        chroma_dir = result[0]
        return Chroma(persist_directory=chroma_dir, embedding_function=embedding_function).as_retriever()
    return None

def update_retriever_status(user_id: str, file_path: str, status: Status):
    cursor.execute("UPDATE retrievers SET status = ? WHERE user_id = ? AND file_path = ?", (status.value, user_id, file_path))
    conn.commit()

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

@app.get("/api/upload_status/")
async def upload_status(file_path: str, user_id: str = "default_user"):
    cursor.execute("SELECT status FROM retrievers WHERE user_id = ? AND file_path = ?", (user_id, f"{FILE_STORE}{file_path}"))
    result = cursor.fetchone()
    if not result:
        return {"status": "No PDF uploaded."}
    return {"file": file_path.lstrip(FILE_STORE), "status": result[0]}


@app.get("/api/get_files/")
async def get_files(user_id: str = "default_user"):
    cursor.execute("SELECT file_path FROM retrievers WHERE user_id = ?", (user_id,))
    result = cursor.fetchall()
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

        save_retriever_to_db(user_id, file_path, chroma_dir)

        background_tasks.add_task(background_chunker, file_path, chroma_dir, user_id)
        
        update_retriever_status(user_id, file_path, Status.PROCESSING)
        
        # parallel_splits = parallel_chunking(file_path) 
        # Chroma.from_documents(
        #     documents=parallel_splits,
        #     embedding=embedding_function,
        #     persist_directory=chroma_dir,
        # )

        # save_retriever_to_db(user_id, file_path, chroma_dir)
        # logging.info(f"Retriever initialized for user: {user_id}")

        return {"message": "PDF uploaded and processing started."}
    except Exception as e:
        logging.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


# Question-answering endpoint
class QuestionRequest(BaseModel):
    question: str


@app.post("/api/ask_question/")
async def ask_question(request: QuestionRequest, file: str, user_id: str = "default_user"):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="PDF file not provided.")
        
        retriever = get_retriever_from_db(user_id, f"{FILE_STORE}{file}")
        if not retriever:
            raise HTTPException(status_code=400, detail="PDF not uploaded.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=False,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )

        response = qa_chain.invoke(request.question)
        return {"answer": response["result"]}
    except Exception as e:
        logging.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


# Global error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, e: HTTPException):
    logging.error(f"HTTP Exception: {e.detail}")
    return JSONResponse(
        status_code=e.status_code, content={"message": e.detail}
    )
