import os
import chromadb
from PyPDF2 import PdfReader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set. Set it in the terminal first.")

llm = ChatGroq(
    temperature=0,
    model="openai/gpt-oss-120b"
)

reader = PdfReader("Document_QA_RAG_More_QA.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

chunk_size = 500

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

client = chromadb.Client()

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)

collection.add(
    documents=chunks,
    ids=[f"id{i}" for i in range(len(chunks))]
)

query = input("Ask a question: ")

results = collection.query(
    query_texts=[query],
    n_results=3
)

context = " ".join(results["documents"][0])

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question using ONLY the document context below.

If the answer is not available in the document, say:
"Information not found in the document."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": query
})

print("\nAnswer:")
print(response.content)