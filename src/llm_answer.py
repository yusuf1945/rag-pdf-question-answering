# src/llm_answer.py

import os
import time
from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=api_key)


def generate_answer(question, retrieved_chunks):
    """
    Generate a final answer using the user question and retrieved PDF chunks.

    Parameters:
        question: User question
        retrieved_chunks: List of dictionaries from retrieval step

    Returns:
        answer: LLM-generated answer grounded in PDF context
    """

    if not api_key:
        return "Error: GEMINI_API_KEY not found. Please add it to your .env file."

    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
Source Chunk {i}
Chunk Number: {item['chunk_index'] + 1}
Similarity Score: {item['similarity_score']:.4f}

{item['chunk_text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a PDF question-answering assistant.

Your job is to answer the user's question using ONLY the provided PDF context.

Rules:
1. Use only the information present in the context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I could not find the answer in the uploaded PDF."
4. Keep the answer clear and helpful.
5. If the user asks for a summary, summarize only the provided context.
6. Mention the relevant chunk numbers at the end.

PDF Context:
{context}

User Question:
{question}

Final Answer:
"""

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response.text

        except Exception as e:
            error_message = str(e)

            if attempt < max_retries - 1:
                time.sleep(2)
                continue

            return (
                "Gemini API is currently unavailable or overloaded. "
                "This is usually temporary. Please try again after a few minutes.\n\n"
                f"Technical details: {error_message}"
            )