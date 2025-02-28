import requests
from bs4 import BeautifulSoup
import ollama
import chromadb
import gradio as gr

def scrape_text(url):
    """Scrapes text content from a given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"Failed to fetch the webpage: {response.status_code}"
    
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs if p.get_text()])
    return text

def embed_text(text):
    """Generates embeddings for the extracted text using Ollama."""
    embeddings_response = ollama.embeddings("all-minilm:l6-v2", text)
    embeddings = embeddings_response["embedding"]
    return embeddings, text

def store_embeddings(embeddings, text):
    """Stores embeddings in a ChromaDB vector database."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("web_scraped_content")
    collection.add(embeddings=[embeddings], documents=[text], ids=["1"])
    return collection

def query_rag(question):
    """Retrieves relevant context from ChromaDB and generates an answer using Ollama."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("web_scraped_content")
    results = collection.query(query_texts=[question], n_results=3)
    if not results["documents"]:
        return "No relevant information found."
    context = "\n".join(results["documents"][0])
    response = ollama.chat(model="smollm:135m", messages=[
        {"role": "system", "content": "You are an assistant providing answers based on retrieved documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ])
    return response["message"]["content"]

def process_url(url):
    """Processes a URL by scraping, embedding, and storing content."""
    text = scrape_text(url)
    if "Failed to fetch" in text:
        return text
    embeddings, processed_text = embed_text(text)
    store_embeddings(embeddings, processed_text)
    return "Data successfully stored in ChromaDB! You can now ask questions."

with gr.Blocks() as demo:
    gr.Markdown("### URL Scraper and RAG-based Q&A")
    url_input = gr.Textbox(label="Enter URL")
    process_button = gr.Button("Scrape and Store")
    status_output = gr.Textbox(label="Status")
    question_input = gr.Textbox(label="Ask a Question")
    answer_output = gr.Textbox(label="Answer")
    ask_button = gr.Button("Get Answer")
    
    process_button.click(process_url, inputs=url_input, outputs=status_output)
    ask_button.click(query_rag, inputs=question_input, outputs=answer_output)

demo.launch()