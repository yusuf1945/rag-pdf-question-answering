import streamlit as st

from src.pdf_loader import extract_text_from_pdf, clean_text
from src.text_splitter import split_text_into_chunks
from src.embeddings import create_embeddings, create_query_embedding
from src.vector_search import get_top_k_chunks
from src.llm_answer import generate_answer

# Page Configuration
def run():
  st.set_page_config(
      page_title="RAG PDF Assistant",
      page_icon="📄",
      layout="wide"
  )

  # Sidebar

  with st.sidebar:
      st.header("About")
      st.write(
          "This app uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded PDF documents."
      )

      st.subheader("Current Version")
      st.info("Level 1 — Basic RAG")

      st.subheader("Features-Level 1")
      st.markdown(
          """
  - PDF upload
  - Text extraction
  - Chunking with overlap
  - Embeddings using sentence-transformers
  - Semantic retrieval using cosine similarity
  - Gemini-based answer generation
  - Retrieved source chunk display
  """
      )

      st.subheader("Next Improvements")
      st.markdown(
          """
  - Multiple PDF upload
  - Page-number citations
  - Chat history
  - Better chunking strategy
  - Deployment
  """
      )

  # Header

  st.title("RAG-based PDF Question Answering System")
  st.caption(
      "Upload a PDF and ask questions grounded in the document using Retrieval-Augmented Generation."
  )

  st.markdown("---")

  # Input Section

  st.subheader("Upload Document")

  uploaded_file = st.file_uploader(
      "Upload your PDF",
      type=["pdf"],
      help="Upload a text-based PDF. Scanned PDFs will be supported later with OCR."
  )

  question = st.text_input(
      "Ask a question from the PDF",
      placeholder="Example: Who wrote this document?"
  )

  st.markdown("Try questions like:")
  st.markdown(
      """
  - Who wrote this document?
  - What is the main objective?
  - Summarize the relevant section.
  - What skills are mentioned?
  - What models or methods are discussed?
  """
  )

  # Main RAG Pipeline

  if uploaded_file is not None:
      st.success("PDF uploaded successfully!")
      st.write(f"**File name:** {uploaded_file.name}")

      progress = st.progress(0, text="Starting PDF processing...")

      # Step 1: Extract text
      progress.progress(15, text="Extracting text from PDF...")
      raw_text = extract_text_from_pdf(uploaded_file)

      # Step 2: Clean text
      progress.progress(30, text="Cleaning extracted text...")
      cleaned_text = clean_text(raw_text)

      if cleaned_text.strip() == "":
          st.error("No text could be extracted. This may be a scanned PDF.")
          st.stop()

      # Step 3: Chunk text
      progress.progress(45, text="Splitting text into chunks...")
      chunk_size = 800
      overlap = 150

      chunks = split_text_into_chunks(
          cleaned_text,
          chunk_size=chunk_size,
          overlap=overlap
      )

      # Step 4: Create embeddings
      progress.progress(65, text="Creating embeddings for chunks...")
      chunk_embeddings = create_embeddings(chunks)

      progress.progress(80, text="Ready for question answering.")

      st.markdown("---")

      # Processing Insights

      st.subheader("🔎 Processing Insights")

      col1, col2, col3, col4 = st.columns(4)

      with col1:
          st.metric("Characters Extracted", len(cleaned_text))

      with col2:
          st.metric("Chunks Created", len(chunks))

      with col3:
          st.metric("Embedding Dimensions", chunk_embeddings.shape[1])

      with col4:
          st.metric("Chunk Size", chunk_size)

      with st.expander("Step 1: Extracted Text Preview"):
          st.write(cleaned_text[:2000])

      with st.expander("Step 2: Chunking Details"):
          st.write(f"Total chunks created: {len(chunks)}")
          st.write(f"Chunk size: {chunk_size}")
          st.write(f"Overlap: {overlap}")

          selected_chunk_number = st.number_input(
              "Preview chunk number",
              min_value=1,
              max_value=len(chunks),
              value=1
          )

          st.write(chunks[selected_chunk_number - 1])

      with st.expander("Step 3: Embedding Details"):
          st.write("Embedding shape:", chunk_embeddings.shape)
          st.write(
              f"This means {chunk_embeddings.shape[0]} chunks were converted into "
              f"{chunk_embeddings.shape[1]}-dimensional vectors."
          )

      # Generate Answer

      st.markdown("---")
      st.subheader("Ask the Document")

      if question.strip() == "":
          st.warning("Enter a question to generate an answer.")
      else:
          if st.button("Generate Answer from PDF", type="primary"):
              progress.progress(85, text="Retrieving relevant chunks...")

              retrieved_chunks = get_top_k_chunks(
                  question=question,
                  chunks=chunks,
                  chunk_embeddings=chunk_embeddings,
                  create_query_embedding=create_query_embedding,
                  top_k=3
              )

              progress.progress(95, text="Generating answer using LLM...")

              answer = generate_answer(question, retrieved_chunks)

              progress.progress(100, text="Answer generated successfully!")

              tab1, tab2, tab3 = st.tabs(
                  ["💡 Answer", "📚 Retrieved Chunks", "🧪 Pipeline Insights"]
              )

              with tab1:
                  st.subheader("💡 Final Answer")
                  st.success(answer)

              with tab2:
                  st.subheader("📚 Retrieved Source Chunks")

                  for rank, result in enumerate(retrieved_chunks, start=1):
                      with st.expander(f"Source Chunk {rank}"):
                          st.write(f"Chunk number: {result['chunk_index'] + 1}")
                          st.write(f"Similarity score: {result['similarity_score']:.4f}")
                          st.write(result["chunk_text"])

              with tab3:
                  st.subheader("🧪 Pipeline Insights")
                  st.write("The app followed this pipeline:")

                  st.markdown(
                      """
  1. PDF uploaded  
  2. Text extracted using `pypdf`  
  3. Text cleaned  
  4. Text split into overlapping chunks  
  5. Chunks converted into embeddings  
  6. User question converted into embedding  
  7. Top relevant chunks retrieved using cosine similarity  
  8. Gemini LLM generated final grounded answer  
  """
                  )

                  st.write("Question:", question)
                  st.write("Total chunks searched:", len(chunks))
                  st.write("Top-k chunks retrieved:", len(retrieved_chunks))

  else:
      st.info("Upload a PDF to begin.")