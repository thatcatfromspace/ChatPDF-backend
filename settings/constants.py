from enum import Enum

class Status(Enum):
    INITIALIZED = "initialized"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

FILE_STORE = "files/"
LANGUAGE_MODEL = "mistral"

RETRIEVER_DB_PATH = "retrievers.db"
RESPONSE_DB_PATH = "responses.db"
