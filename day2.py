import os
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set. Set it in the terminal first.")

llm = ChatGroq(
    temperature=0,
    model="openai/gpt-oss-20b"
)

client = chromadb.Client()

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="jobs_collection",
    embedding_function=embedding_function
)

collection.add(
    documents=[
        "Machine learning and Python AI solutions",
        "WordPress website development services",
        "Magento e-commerce platform development"
    ],
    metadatas=[
        {"links": "https://example.com/ml-python-portfolio"},
        {"links": "https://example.com/wordpress-portfolio"},
        {"links": "https://example.com/magento-portfolio"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

json_res = [{
    "title": "AI Engineer",
    "skills": "Python, Machine Learning, NLP, APIs",
    "description": "Hiring AI engineer to build ML models and NLP systems."
}]

job = json_res[0]

raw_links = collection.query(
    query_texts=[job["skills"]],
    n_results=2
)["metadatas"]

clean_links = [
    item["links"]
    for group in raw_links
    for item in group
]

unique_links = list(set(clean_links))

prompt_email = PromptTemplate.from_template("""
### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are Mohan, a business development executive at AtliQ.
AtliQ is an AI & Software Consulting company.

Write a cold email to the client describing AtliQ's
capability to fulfill their requirements.

Also include the most relevant portfolio links:
{link_list}

Remember you are Mohan, BDE at AtliQ.
Do not provide a preamble.

### EMAIL:
""")

chain_email = prompt_email | llm

res = chain_email.invoke({
    "job_description": str(job),
    "link_list": unique_links
})

print(res.content)