# src/embeddings.py

from sentence_transformers import SentenceTransformer


# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    """
    Convert text chunks into numerical embedding vectors.

    Parameters:
        chunks: List of text chunks

    Returns:
        embeddings: NumPy array of shape
        (number_of_chunks, embedding_dimension)
    """

    embeddings = model.encode(chunks)

    return embeddings


def create_query_embedding(question):
    """
    Convert user question into a numerical embedding vector.

    Parameters:
        question: User question as text

    Returns:
        query_embedding: NumPy array of shape (embedding_dimension,)
    """

    query_embedding = model.encode(question)

    return query_embedding