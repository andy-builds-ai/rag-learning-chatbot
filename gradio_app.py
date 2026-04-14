"""
RAG Learning Chatbot - Gradio Web Interface
"""

import os
import gradio as gr
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "./faiss_index"
MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
RETRIEVER_K = 6

SYSTEM_PROMPT = """Du bist ein Lern-Assistent. Beantworte die Frage AUSSCHLIESSLICH mit Informationen aus dem Kontext.
Erfinde NICHTS dazu. Nutze KEIN eigenes Wissen.
Wenn der Kontext die Frage nicht beantwortet, sage nur: Ich finde dazu nichts im Wissen.
Antworte auf Deutsch.

Kontext:
{kontext}

Frage: {frage}"""

llm = OllamaLLM(model=MODEL_NAME)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})


def ask(frage, history):
    results = retriever.invoke(frage)
    context = "\n".join([doc.page_content for doc in results])

    sources = set()
    for doc in results:
        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        sources.add(filename)

    prompt = SYSTEM_PROMPT.format(kontext=context, frage=frage)
    answer = llm.invoke(prompt)

    return f"{answer}\n\nQuellen: {', '.join(sources)}"


demo = gr.ChatInterface(
    fn=ask,
    title="RAG Learning Chatbot",
    description="AI Engineering Wissen — lokal mit Ollama + LangChain + FAISS",
    examples=["Was ist RAG?", "Wie funktioniert ein LLM?", "Was ist Prompt Engineering?"],
)

if __name__ == "__main__":
    demo.launch()