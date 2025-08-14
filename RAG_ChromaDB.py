import chromadb
from chromadb.utils import embedding_functions
from wikipediaapi import Wikipedia
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client_openAI = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

COLLECTION_NAME = "Nguyen_Nhat_Anh"
client = chromadb.PersistentClient(path="./data")


embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function)

wiki = Wikipedia("HocCodeAI/0.0", 'en')
doc = wiki.page("Nguyen_Nhat_Anh").text

# print(doc)

paragraphs = doc.split("\n\n")
for i, paragraph in enumerate(paragraphs):
    collection.add(
        documents=[paragraph],
        ids=[str(i)])

query = "What is Nguyen Nhat Anh most famous for?"
context = collection.query(
    query_texts=[query],
    n_results=3
)["documents"][0]

prompt = f"""
Use the following CONTEXT to answer the QUESTION at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use an unbiased and journalistic tone.

CONTEXT: {context}

QUESTION: {query}
"""

# print(prompt)

response = client_openAI.chat.completions.create(
    model="gemma2-9b-it",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

print(response.choices[0].message.content.strip())
