# RAG_with_ChromaDB_and_OpenAI

RAG_basic_with_ChromaDB_and_OpenAI is a simple implementation of Retrieval-Augmented Generation (RAG) that combines:

ChromaDB → A local or persistent vector database used to store and search document embeddings.

OpenAI API → A powerful language model for generating human-like responses.

How it works:

Document Ingestion → Load documents/text, split them into chunks, and create embeddings using OpenAI’s embedding model.

Store in ChromaDB → Save the text chunks and their embeddings in ChromaDB for fast similarity search.

Query Processing → When a user asks a question, convert it into an embedding and search ChromaDB for the most relevant chunks.

Augmented Response → Pass the retrieved chunks along with the query to OpenAI’s GPT model to generate an accurate, context-aware answer.