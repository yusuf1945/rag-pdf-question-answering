import streamlit as st
import time
from src.pdf_loader import clean_text, extract_pages_from_pdf
from src.text_splitter import split_text_into_chunks
from src.embeddings import create_embeddings, create_query_embedding
from src.vector_search import get_top_k_chunks
from src.llm_answer import generate_answer


def run():
    st.set_page_config(
        page_title="RAG PDF Assistant",
        page_icon="📄",
        layout="wide"
    )

    # Sidebar
    with st.sidebar:
        st.title("RAG Controls")

        st.markdown("### App Info")
        st.info("Level 2 — Multi-PDF Conversational RAG")

        st.markdown("### Features")
        st.markdown(
            """
            - Multi-PDF upload
            - Page-wise extraction
            - Source + page citations
            - Top-K retrieval control
            - Chunk size / overlap control
            - PDF metadata filtering
            - Conversational memory
            - Performance metrics
            """
        )

        st.markdown("---")

        st.markdown("### Session")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Built with Streamlit · Embeddings · Gemini · RAG")

    # Header
    st.title("RAG PDF Assistant — Level 2")
    st.caption(
        "Ask grounded questions across multiple PDFs with citations, memory, and retrieval controls."
    )

    # Chat history memory initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("---")

    # Upload Section
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.expander("📁 Uploaded Files", expanded=True):
            for file in uploaded_files:
                st.write(f"✅ {file.name}")

    question = st.text_input(
        "Ask a question from the PDFs",
        placeholder="Example: What is the main objective?"
    )

    # Retrieval and Chunking Controls
    st.markdown("### Retrieval & Chunking Settings")

    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=3
        )

    with control_col2:
        chunk_size = st.slider(
            "Chunk Size",
            min_value=300,
            max_value=1500,
            value=800,
            step=100
        )

    with control_col3:
        overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=400,
            value=150,
            step=50
        )

    if overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
        st.stop()

    # Main RAG Pipeline
    if uploaded_files is not None and len(uploaded_files) > 0:
        st.success("PDFs uploaded successfully!")

        all_chunks = []
        all_metadata = []
        total_characters = 0
        all_text_preview = ""

        preprocessing_start_time = time.perf_counter()

        extraction_time = 0
        cleaning_chunking_time = 0
        embedding_time = 0

        progress = st.progress(0, text="Starting PDF processing...")

        for uploaded_file in uploaded_files:
            st.write(f"**Processing:** {uploaded_file.name}")

            # Step 1: Extract pages
            progress.progress(15, text=f"Extracting pages from {uploaded_file.name}...")

            start_time = time.perf_counter()
            pages_data = extract_pages_from_pdf(uploaded_file)
            extraction_time += time.perf_counter() - start_time

            pdf_has_text = False

            # Step 2 and 3: Clean and chunk page-wise
            for page_data in pages_data:
                page_number = page_data["page"]
                raw_text = page_data["text"]

                progress.progress(
                    30,
                    text=f"Cleaning text from {uploaded_file.name}, page {page_number}..."
                )

                start_time = time.perf_counter()

                cleaned_text = clean_text(raw_text)

                if cleaned_text.strip() == "":
                    continue

                pdf_has_text = True
                total_characters += len(cleaned_text)

                all_text_preview += f"\n\n--- {uploaded_file.name} | Page {page_number} ---\n"
                all_text_preview += cleaned_text[:500]

                progress.progress(
                    45,
                    text=f"Splitting {uploaded_file.name}, page {page_number} into chunks..."
                )

                chunks = split_text_into_chunks(
                    cleaned_text,
                    chunk_size=chunk_size,
                    overlap=overlap
                )

                cleaning_chunking_time += time.perf_counter() - start_time

                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": uploaded_file.name,
                        "page": page_number
                    })

            if not pdf_has_text:
                st.warning(f"No text extracted from {uploaded_file.name}. Skipping.")
                continue

        # Step 4: Create embeddings
        if len(all_chunks) == 0:
            st.error("No text extracted from any PDF.")
            st.stop()

        if len(all_chunks) > 500:
            st.warning(
                f"Large document detected: {len(all_chunks)} chunks created. "
                "Embedding may take time. Consider increasing chunk size or uploading fewer PDFs."
            )

        progress.progress(65, text="Creating embeddings for all PDFs...")

        try:
            start_time = time.perf_counter()
            chunk_embeddings = create_embeddings(all_chunks)
            embedding_time = time.perf_counter() - start_time

        except Exception as e:
            st.error("Embedding creation failed. Please try again or upload a smaller/cleaner PDF.")
            st.write(f"Technical details: {e}")
            st.stop()

        preprocessing_total_time = time.perf_counter() - preprocessing_start_time

        progress.progress(80, text="Ready for question answering.")

        st.markdown("---")

        # Processing Insights
        st.subheader("Document Processing Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Characters", total_characters)

        with col2:
            st.metric("Chunks", len(all_chunks))

        with col3:
            st.metric("Embedding Dim", chunk_embeddings.shape[1])

        with col4:
            st.metric("Chunk Size", chunk_size)

        with col5:
            st.metric("Overlap", overlap)

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

        with perf_col1:
            st.metric("Extraction", f"{extraction_time:.2f}s")

        with perf_col2:
            st.metric("Clean + Chunk", f"{cleaning_chunking_time:.2f}s")

        with perf_col3:
            st.metric("Embedding", f"{embedding_time:.2f}s")

        with perf_col4:
            st.metric("Total Prep", f"{preprocessing_total_time:.2f}s")

        with st.expander("Extracted Text Preview"):
            st.write(all_text_preview)

        with st.expander("Chunk Preview & Metadata"):
            st.write(f"Total chunks created: {len(all_chunks)}")
            st.write(f"Chunk size: {chunk_size}")
            st.write(f"Overlap: {overlap}")

            selected_chunk_number = st.number_input(
                "Preview chunk number",
                min_value=1,
                max_value=len(all_chunks),
                value=1
            )

            selected_index = selected_chunk_number - 1
            selected_metadata = all_metadata[selected_index]

            st.write(f"Source PDF: {selected_metadata['source']}")
            st.write(f"Page number: {selected_metadata['page']}")
            st.write(all_chunks[selected_index])

        with st.expander("Embedding Details"):
            st.write("Embedding shape:", chunk_embeddings.shape)
            st.write(
                f"This means {chunk_embeddings.shape[0]} chunks were converted into "
                f"{chunk_embeddings.shape[1]}-dimensional vectors."
            )

        # Metadata Filtering by PDF
        st.markdown("---")
        st.subheader("Search Scope")

        pdf_options = ["All PDFs"] + sorted(list(set(
            metadata["source"] for metadata in all_metadata
        )))

        selected_pdf = st.selectbox(
            "Search inside",
            pdf_options
        )

        if selected_pdf == "All PDFs":
            search_chunks = all_chunks
            search_metadata = all_metadata
            search_embeddings = chunk_embeddings
        else:
            selected_indices = [
                i for i, metadata in enumerate(all_metadata)
                if metadata["source"] == selected_pdf
            ]

            search_chunks = [all_chunks[i] for i in selected_indices]
            search_metadata = [all_metadata[i] for i in selected_indices]
            search_embeddings = chunk_embeddings[selected_indices]

        scope_col1, scope_col2 = st.columns(2)

        with scope_col1:
            st.info(f"Searching in: {selected_pdf}")

        with scope_col2:
            st.metric("Chunks in Scope", len(search_chunks))

        if len(search_chunks) == 0:
            st.error("No chunks available in the selected search scope.")
            st.stop()

        # Generate Answer
        st.markdown("---")
        st.subheader("💬 Ask the Documents")

        st.subheader("🗨️ Conversation")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question.strip() == "":
            st.warning("Enter a question to generate an answer.")
        else:
            if st.button("Generate Answer", type="primary"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                # Building short-term conversational memory
                recent_history = st.session_state.messages[-5:-1]

                history_text = ""
                for msg in recent_history:
                    history_text += f"{msg['role']}: {msg['content']}\n"

                memory_enhanced_question = f"""
Previous conversation:
{history_text}

Current question:
{question}
"""

                progress.progress(85, text="Retrieving relevant chunks...")

                effective_top_k = min(top_k, len(search_chunks))

                try:
                    retrieval_start_time = time.perf_counter()

                    retrieved_chunks = get_top_k_chunks(
                        question=memory_enhanced_question,
                        chunks=search_chunks,
                        chunk_embeddings=search_embeddings,
                        create_query_embedding=create_query_embedding,
                        top_k=effective_top_k
                    )

                    retrieval_time = time.perf_counter() - retrieval_start_time

                except Exception as e:
                    st.error("Retrieval failed. Please try changing the search scope or reducing chunk settings.")
                    st.write(f"Technical details: {e}")
                    st.stop()

                if len(retrieved_chunks) == 0:
                    st.warning("No relevant chunks were retrieved. Try increasing Top-K or selecting All PDFs.")
                    st.stop()

                progress.progress(95, text="Generating answer using LLM...")

                llm_start_time = time.perf_counter()
                answer = generate_answer(memory_enhanced_question, retrieved_chunks)
                llm_time = time.perf_counter() - llm_start_time

                answer_total_time = retrieval_time + llm_time

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                progress.progress(100, text="Answer generated successfully!")

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Final Answer", "Sources", "Pipeline", "Performance"]
                )

                with tab1:
                    st.subheader("Final Answer")
                    st.markdown(answer)

                    st.markdown("### Sources Used")

                    used_sources = set()

                    for result in retrieved_chunks:
                        metadata = search_metadata[result["chunk_index"]]
                        used_sources.add((metadata["source"], metadata["page"]))

                    for source, page in sorted(used_sources):
                        st.write(f"- {source}, Page {page}")

                with tab2:
                    st.subheader("Retrieved Source Chunks")

                    for rank, result in enumerate(retrieved_chunks, start=1):
                        metadata = search_metadata[result["chunk_index"]]

                        with st.expander(f"Source Chunk {rank}"):
                            st.write(f"Source PDF: {metadata['source']}")
                            st.write(f"Page number: {metadata['page']}")
                            st.write(f"Chunk number: {result['chunk_index'] + 1}")
                            st.write(f"Similarity score: {result['similarity_score']:.4f}")
                            st.write(result["chunk_text"])

                with tab3:
                    st.subheader("Pipeline Insights")
                    st.write("The app followed this pipeline:")

                    st.markdown(
                        """
                        1. PDFs uploaded  
                        2. Text extracted page-by-page using `pypdf`  
                        3. Text cleaned page-wise  
                        4. Text split into overlapping chunks  
                        5. Each chunk stored with source PDF and page number  
                        6. Chunks converted into embeddings  
                        7. User question + recent chat history converted into a memory-enhanced query  
                        8. Search scope selected using PDF metadata filtering  
                        9. Top relevant chunks retrieved using cosine similarity  
                        10. Gemini LLM generated final grounded answer  
                        """
                    )

                    st.write("Question:", question)
                    st.write("Search scope:", selected_pdf)
                    st.write("Total chunks searched:", len(search_chunks))
                    st.write("Top-K requested:", top_k)
                    st.write("Effective Top-K used:", effective_top_k)
                    st.write("Top-K chunks retrieved:", len(retrieved_chunks))

                with tab4:
                    st.subheader("⏱️ Performance Metrics")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric("Retrieval Time", f"{retrieval_time:.2f}s")

                    with metric_col2:
                        st.metric("LLM Generation Time", f"{llm_time:.2f}s")

                    with metric_col3:
                        st.metric("Answer Pipeline Time", f"{answer_total_time:.2f}s")

                    st.markdown("---")
                    st.write(f"PDF extraction time: {extraction_time:.2f} seconds")
                    st.write(f"Cleaning + chunking time: {cleaning_chunking_time:.2f} seconds")
                    st.write(f"Embedding time: {embedding_time:.2f} seconds")
                    st.write(f"Preprocessing total time: {preprocessing_total_time:.2f} seconds")

    else:
        st.info("Upload a PDF to begin.")