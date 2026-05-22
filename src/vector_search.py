# src/vector_search.py

import numpy as np


def cosine_similarity(query_embedding, chunk_embeddings):
    """
    Calculate cosine similarity between the query embedding
    and all chunk embeddings.

    Parameters:
        query_embedding: Vector of user question
        chunk_embeddings: Vectors of all PDF chunks

    Returns:
        similarities: Similarity score for each chunk
    """

    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

    dot_products = np.dot(chunk_embeddings, query_embedding)

    similarities = dot_products / (chunk_norms * query_norm)

    return similarities


def get_top_k_chunks(question, chunks, chunk_embeddings, create_query_embedding, top_k=3):
    """
    Retrieve top-k most relevant chunks for a user question.

    Parameters:
        question: User question
        chunks: List of text chunks
        chunk_embeddings: Embeddings of all chunks
        create_query_embedding: Function to convert question into embedding
        top_k: Number of relevant chunks to retrieve

    Returns:
        results: List of dictionaries containing chunk, index, and score
    """

    query_embedding = create_query_embedding(question)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk_index": int(index),
            "chunk_text": chunks[index],
            "similarity_score": float(similarities[index])
        })

    return results