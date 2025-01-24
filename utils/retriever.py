from sqlite3 import Connection, Cursor
from settings.constants import Status
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class Retriever:
    def __init__(self, retriever_conn: Connection, retriever_cursor: Cursor, embedding_function: OllamaEmbeddings):
        self.retriever_conn: Connection = retriever_conn
        self.retriever_cursor: Cursor = retriever_cursor
        self.embedding_function: OllamaEmbeddings = embedding_function
     
    def save_retriever(self, user_id: str, file_path: str, chroma_dir: str) -> None:
        self.retriever_cursor.execute(
            "INSERT INTO retrievers (user_id, file_path, chroma_dir) VALUES (?, ?, ?)",
            (user_id, file_path, chroma_dir),
        )
        self.retriever_conn.commit()

    def get_retriever(self, user_id: str, file_path: str):
        self.retriever_cursor.execute("SELECT chroma_dir, status FROM retrievers WHERE user_id = ? AND file_path = ?", (user_id, file_path,))
        result = self.retriever_cursor.fetchone()
        if result and result[1] == Status.COMPLETED.value:
            chroma_dir = result[0]
            return Chroma(persist_directory=chroma_dir, embedding_function=self.embedding_function).as_retriever()
        return None


    def update_retriever_status(self, user_id: str, file_path: str, status: Status) -> None:
        self.retriever_cursor.execute("UPDATE retrievers SET status = ? WHERE user_id = ? AND file_path = ?", (status.value, user_id, file_path))
        self.retriever_conn.commit()