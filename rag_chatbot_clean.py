"""
RAG Learning Chatbot - Local AI-powered Q&A system for AI Engineering study.
Uses Ollama (Llama 3.2) + LangChain + FAISS for retrieval-augmented generation.
Knowledge base stored as text files in the 'wissen/' directory.
"""

import os
import json
from datetime import datetime
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

INDEX_DIR = "./faiss_index"
WISSEN_DIR = "wissen"
STATUS_FILE = "./index_status.json"
MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 6

SYSTEM_PROMPT = """Du bist ein Lern-Assistent. Beantworte die Frage AUSSCHLIESSLICH mit Informationen aus dem Kontext.
Erfinde NICHTS dazu. Nutze KEIN eigenes Wissen.
Wenn der Kontext die Frage nicht beantwortet, sage nur: Ich finde dazu nichts im Wissen.
Antworte auf Deutsch.

Kontext:
{kontext}

Frage: {frage}"""


def load_documents():
    loader = DirectoryLoader(
        WISSEN_DIR, glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


def build_index(documents, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    status = {
        "letzter_build": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "anzahl_dateien": len(documents),
        "anzahl_chunks": len(chunks)
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

    return vectorstore, status


def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return None


def show_help():
    print("\n--- Commands ---")
    print("help          - Show this help")
    print("status        - Show index info (files, chunks, last build)")
    print("neues wissen  - Rebuild index (after adding new files)")
    print("exit          - Quit the chatbot\n")


def show_status():
    status = load_status()
    if status:
        print(f"\nFiles:      {status['anzahl_dateien']}")
        print(f"Chunks:     {status['anzahl_chunks']}")
        print(f"Last build: {status['letzter_build']}\n")
    else:
        print("\nNo index found. Type 'neues wissen' to build one.\n")


def ask_question(frage, retriever, llm):
    results = retriever.invoke(frage)
    context = "\n".join([doc.page_content for doc in results])

    sources = set()
    for doc in results:
        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        sources.add(filename)

    prompt = SYSTEM_PROMPT.format(kontext=context, frage=frage)
    answer = llm.invoke(prompt)

    print(f"\nAntwort: {answer}")
    print(f"Quellen: {', '.join(sources)}\n")


def main():
    llm = OllamaLLM(model=MODEL_NAME)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    documents = load_documents()
    print(f"{len(documents)} documents loaded.")

    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(
            INDEX_DIR, embeddings,
            allow_dangerous_deserialization=True
        )
        print("Existing index loaded.")
    else:
        vectorstore, _ = build_index(documents, embeddings)
        print("New index created.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    print("RAG Chatbot ready! Type 'help' for commands.\n")

    while True:
        frage = input("Your question: ")

        if frage.lower() == "exit":
            print("Goodbye!")
            break
        if frage.lower() == "help":
            show_help()
            continue
        if frage.lower() == "status":
            show_status()
            continue
        if frage.lower() == "neues wissen":
            print("Reloading documents...")
            documents = load_documents()
            vectorstore, status = build_index(documents, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
            print(f"Index rebuilt: {status['anzahl_dateien']} files, {status['anzahl_chunks']} chunks.\n")
            continue

        ask_question(frage, retriever, llm)


if __name__ == "__main__":
    main()