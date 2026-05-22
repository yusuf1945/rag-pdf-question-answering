# src/text_splitter.py

def split_text_into_chunks(text, chunk_size=800, overlap=150):
    """
    Split long text into smaller overlapping chunks.

    Parameters:
        text: Full extracted PDF text
        chunk_size: Number of characters in each chunk
        overlap: Number of characters repeated between chunks

    Returns:
        chunks: List of text chunks
    """

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks