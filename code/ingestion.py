import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, PROJECT_ROOT

load_dotenv()

collections = ["claude", "hackerrank", "visa"]

embeddings = OpenAIEmbeddings(chunk_size=50)

_vectorstores = {}

def get_collection_chroma_dir(collection):
    return PROJECT_ROOT / f".chroma-{collection}"

def get_collection_path(collection):
    return os.path.join(DATA_DIR, collection)

def init_vectorstores():
    for collection in collections:
        _vectorstores[collection] = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=get_collection_chroma_dir(collection),
        )

init_vectorstores()

def get_vectorstore(collection):
    return _vectorstores[collection]


def ingest_collection(collection):
    docs = []
    collection_path = get_collection_path(collection)
    for root, dirs, files in os.walk(collection_path):
        for file in files:
            full_path = os.path.join(root, file)
            # print(full_path)
            # print(f"metadata: {os.path.relpath(full_path, collection_path)}")
            doc_source = os.path.relpath(full_path, collection_path)
            orig_docs = TextLoader(full_path, encoding="UTF-8").load()
            for orig_doc in orig_docs:
                docs.append(Document(page_content=orig_doc.page_content, metadata={"source": doc_source}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)
    print(f"Embedding {len(splits)} split documents")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=get_collection_chroma_dir(collection),
        collection_name=collection,
    )


if __name__ == "__main__":
    for collection in collections:
        ingest_collection(collection)
