from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.chunker import build_chunks
from app.config import LOCAL_EMBEDDING_MODEL_NAME, VECTORSTORE_DIR


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL_NAME)


def ensure_vectorstore_dir(vectorstore_dir: Path | None = None) -> Path:
    base_dir = vectorstore_dir or VECTORSTORE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def build_vectorstore() -> FAISS:
    chunks = build_chunks()
    print(f"Chunks loaded: {len(chunks)}")

    embeddings = get_embeddings()
    print(f"Local embedding model loaded: {LOCAL_EMBEDDING_MODEL_NAME}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS vectorstore created")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, vectorstore_dir: Path | None = None) -> Path:
    base_dir = ensure_vectorstore_dir(vectorstore_dir)
    vectorstore.save_local(str(base_dir))
    print(f"FAISS vectorstore saved in {base_dir}")
    return base_dir


def vectorstore_exists(vectorstore_dir: Path | None = None) -> bool:
    base_dir = vectorstore_dir or VECTORSTORE_DIR
    index_file = base_dir / "index.faiss"
    store_file = base_dir / "index.pkl"
    return index_file.exists() and store_file.exists()


def load_vectorstore(vectorstore_dir: Path | None = None) -> FAISS:
    base_dir = vectorstore_dir or VECTORSTORE_DIR
    if not vectorstore_exists(base_dir):
        raise FileNotFoundError(f"FAISS vectorstore not found in {base_dir}.")

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(base_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def append_documents_to_vectorstore(
    documents: list[Document],
    vectorstore_dir: Path | None = None,
) -> Path:
    if not documents:
        raise ValueError("No documents were provided for indexing.")

    base_dir = ensure_vectorstore_dir(vectorstore_dir)
    embeddings = get_embeddings()

    if vectorstore_exists(base_dir):
        vectorstore = load_vectorstore(base_dir)
        vectorstore.add_documents(documents)
        print(f"Appended {len(documents)} chunk(s) to existing FAISS vectorstore")
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print(f"FAISS vectorstore created with {len(documents)} chunk(s)")

    vectorstore.save_local(str(base_dir))
    print(f"FAISS vectorstore saved in {base_dir}")
    return base_dir


def main() -> None:
    vectorstore = build_vectorstore()
    save_vectorstore(vectorstore)


if __name__ == "__main__":
    main()
