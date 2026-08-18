import os
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",
    temperature=0
)

response = llm.invoke("The first person to land on the moon was...")
print(response.content)