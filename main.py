import concurrent.futures
import os
import logging
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException
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

cursor.execute("""
CREATE TABLE IF NOT EXISTS retrievers (
    user_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    chroma_dir TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
)
""")
conn.commit()

# App setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    base_url="http://localhost:11434",
    model="tinyllama",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
embedding_function = OllamaEmbeddings(base_url="http://localhost:11434", model="tinyllama")


# Save and retrieve retriever from database
def save_retriever_to_db(user_id: str, file_path: str, chroma_dir: str):
    cursor.execute(
        "REPLACE INTO retrievers (user_id, file_path, chroma_dir) VALUES (?, ?, ?)",
        (user_id, file_path, chroma_dir),
    )
    conn.commit()


def get_retriever_from_db(user_id: str):
    cursor.execute("SELECT chroma_dir FROM retrievers WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    if result:
        chroma_dir = result[0]
        return Chroma(persist_directory=chroma_dir, embedding_function=embedding_function).as_retriever()
    return None


@app.get("/")
async def root():
    return {"message": "Welcome to the PDF QA API!"}

# Helper function 
def process_chunking(document_part):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_splits = text_splitter.split_documents(document_part)
    return all_splits

# Parallelizing the chunking process. Chunking is a CPU-bound operation and can be parallelized to improve performance, 
# because chunks are independent of each other. We split the document into 4 parts and processes each part in parallel 
# using ThreadPoolExecutor and store the results into the vector database.
def parallel_chunking(file_path):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    data = loader.load()

    num_chunks = len(data)  
    num_parts = 4 
    chunk_size = max(1, num_chunks // num_parts) # max() used because fails if num_chunks < num_parts

    document_parts = [data[i:i + chunk_size] for i in range(0, num_chunks, chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parts) as executor:
        results = list(executor.map(process_chunking, document_parts))

    parallel_splits = [chunk for result in results for chunk in result]
    return parallel_splits

# Move PDF chunking to background - better user experience and response times
# Did not provide significant performance improvement - only optimized for large documents.
# def background_chunker(file_path, chroma_dir, user_id):
#     try:
#         parallel_splits = parallel_chunking(file_path) 
#         Chroma.from_documents(
#             documents=parallel_splits,
#             embedding=embedding_function,
#             persist_directory=chroma_dir,
#         )

#         save_retriever_to_db(user_id, file_path, chroma_dir)
#         logging.info(f"Retriever successfully initialized for user: {user_id}")
#     except Exception as e:
#         logging.error(f"Error processing PDF in background: {str(e)}")


# File upload endpoint
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), user_id: str = "default_user"):
    try:
        file_path = f"files/{file.filename}"
        chroma_dir = f"chroma_store/{user_id}"
        os.makedirs('files', exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        parallel_splits = parallel_chunking(file_path) 
        Chroma.from_documents(
            documents=parallel_splits,
            embedding=embedding_function,
            persist_directory=chroma_dir,
        )

        save_retriever_to_db(user_id, file_path, chroma_dir)
        logging.info(f"Retriever initialized for user: {user_id}")

        return {"message": "PDF uploaded and retriever initialized."}
    except Exception as e:
        logging.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


# Question-answering endpoint
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask_question/")
async def ask_question(request: QuestionRequest, user_id: str = "default_user"):
    try:
        retriever = get_retriever_from_db(user_id)
        if not retriever:
            raise HTTPException(status_code=400, detail="PDF not uploaded or retriever not initialized.")

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
async def http_exception_handler(request, exc: HTTPException):
    logging.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code, content={"message": exc.detail}
    )
