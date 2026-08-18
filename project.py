import os
import streamlit as st
import chromadb

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

llm = ChatGroq(
    temperature=0,
    model="openai/gpt-oss-120b",
    groq_api_key=os.environ["GROQ_API_KEY"]
)
client = chromadb.Client()

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)


def ingest_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    return text


st.title("Document QA with RAG")

st.markdown(
    "Upload a PDF document and ask questions about its content."
)

uploaded_files = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        text = ingest_pdf(file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        ids = [
            f"{file.name}_{i}"
            for i in range(len(chunks))
        ]

        try:

            collection.add(
                documents=chunks,
                ids=ids
            )

            st.success(
                f"{file.name} ingested successfully!"
            )

        except Exception:

            st.info(
                f"{file.name} is already loaded."
            )


user_query = st.text_input(
    "Ask a question about the document:"
)


if st.button("Get Advice") and user_query:


    vector_results = collection.query(
        query_texts=[user_query],
        n_results=5
    )

    vector_docs = vector_results["documents"][0]

    keywords = user_query.lower().split()

    keyword_docs = [
        doc
        for doc in vector_docs
        if any(
            keyword in doc.lower()
            for keyword in keywords
        )
    ]

    hybrid_docs = list(
        dict.fromkeys(
            vector_docs + keyword_docs
        )
    )

    docs_text = ""

    for i, doc in enumerate(hybrid_docs, 1):

        docs_text += (
            f"\nDOCUMENT {i}:\n{doc}\n"
        )


    rerank_prompt = PromptTemplate.from_template(
        """
User Query:
{query}

Documents:
{docs}

Rank these documents from most relevant to least relevant
for answering the user's query.

Return ONLY the document numbers in ranked order.

Example:
3, 1, 2, 5, 4
"""
    )


    rerank_chain = rerank_prompt | llm

    reranked_output = rerank_chain.invoke(
        {
            "query": user_query,
            "docs": docs_text
        }
    )


    ranking = reranked_output.content


    ranked_numbers = []

    for word in ranking.replace(",", " ").split():

        try:

            number = int(word)

            if (
                1 <= number <= len(hybrid_docs)
                and number not in ranked_numbers
            ):
                ranked_numbers.append(number)

        except ValueError:
            pass


    if not ranked_numbers:
        ranked_numbers = list(
            range(1, len(hybrid_docs) + 1)
        )
    for number in range(1, len(hybrid_docs) + 1):

        if number not in ranked_numbers:
            ranked_numbers.append(number)

    ranked_docs = [
        hybrid_docs[number - 1]
        for number in ranked_numbers
    ]

    top_context = ranked_docs[:3]

    context = "\n\n".join(top_context)


    final_prompt = PromptTemplate.from_template(
        """
You are a helpful assistant.

Answer the question using ONLY the document context below.

If the answer is not available in the document,
say "I don't know".

DOCUMENT CONTEXT:
{context}

USER QUERY:
{query}

ANSWER:
"""
    )


    rag_chain = final_prompt | llm

    rag_output = rag_chain.invoke(
        {
            "context": context,
            "query": user_query
        }
    )

    st.subheader("Top Retrieved Context")

    for i, doc in enumerate(top_context, 1):

        st.write(f"**Document {i}:**")
        st.write(doc)


    st.subheader("Personalized Answer")

    st.write(rag_output.content)