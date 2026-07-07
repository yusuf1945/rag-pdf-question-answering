import time
import streamlit as st

from src.pdf_loader import clean_text, extract_pages_from_pdf
from src.text_splitter import split_text_into_chunks
from src.embeddings import create_embeddings, create_query_embedding
from src.llm_answer import generate_answer
from src.chroma_store import (
    count_records,
    create_index_id,
    create_pdf_hash,
    index_exists,
    list_indexed_versions,
    search_chroma,
    store_chunks,
)


# IMPORTANT:
# Replace this label with the actual embedding model used inside src/embeddings.py.
# Do not mix vectors from different embedding models inside one collection.
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"


def _build_scope_options():
    versions = list_indexed_versions()

    option_to_filter = {
        "All indexed PDFs": None,
    }

    for version in versions:
        short_hash = version["pdf_hash"][:8]
        label = (
            f"{version['source']} | chunk={version['chunk_size']} | "
            f"overlap={version['overlap']} | {short_hash}"
        )
        option_to_filter[label] = {"index_id": version["index_id"]}

    return option_to_filter


def _index_uploaded_files(uploaded_files, chunk_size, overlap):
    all_new_chunks = []
    all_new_metadata = []
    all_new_ids = []

    total_characters = 0
    all_text_preview = ""

    extraction_time = 0.0
    cleaning_chunking_time = 0.0
    embedding_time = 0.0

    newly_indexed_files = []
    already_indexed_files = []
    failed_files = []

    progress = st.progress(0, text="Starting persistent PDF indexing...")
    preprocessing_start_time = time.perf_counter()

    total_files = len(uploaded_files)

    for file_position, uploaded_file in enumerate(uploaded_files, start=1):
        progress_value = min(10 + int((file_position / total_files) * 25), 35)
        progress.progress(
            progress_value,
            text=f"Checking {uploaded_file.name}...",
        )

        file_bytes = uploaded_file.getvalue()
        pdf_hash = create_pdf_hash(file_bytes)

        index_id = create_index_id(
            pdf_hash=pdf_hash,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model_id=EMBEDDING_MODEL_ID,
        )

        if index_exists(index_id):
            already_indexed_files.append(uploaded_file.name)
            continue

        uploaded_file.seek(0)

        try:
            extraction_start = time.perf_counter()
            pages_data = extract_pages_from_pdf(uploaded_file)
            extraction_time += time.perf_counter() - extraction_start
        except Exception as error:
            failed_files.append((uploaded_file.name, str(error)))
            continue

        pdf_has_text = False
        pdf_chunk_counter = 0
        file_chunks = []
        file_metadata = []
        file_ids = []

        for page_data in pages_data:
            page_number = page_data["page"]
            raw_text = page_data["text"]

            clean_chunk_start = time.perf_counter()
            cleaned_text = clean_text(raw_text)

            if cleaned_text.strip() == "":
                cleaning_chunking_time += time.perf_counter() - clean_chunk_start
                continue

            pdf_has_text = True
            total_characters += len(cleaned_text)

            all_text_preview += (
                f"\n\n--- {uploaded_file.name} | Page {page_number} ---\n"
            )
            all_text_preview += cleaned_text[:500]

            chunks = split_text_into_chunks(
                cleaned_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            cleaning_chunking_time += time.perf_counter() - clean_chunk_start

            for chunk in chunks:
                chunk_record_id = f"{index_id}_chunk_{pdf_chunk_counter}"

                file_chunks.append(chunk)
                file_ids.append(chunk_record_id)
                file_metadata.append(
                    {
                        "source": uploaded_file.name,
                        "page": int(page_number),
                        "pdf_hash": pdf_hash,
                        "index_id": index_id,
                        "chunk_id": int(pdf_chunk_counter),
                        "chunk_size": int(chunk_size),
                        "overlap": int(overlap),
                        "embedding_model": EMBEDDING_MODEL_ID,
                        "source_type": "pdf_text",
                    }
                )

                pdf_chunk_counter += 1

        if not pdf_has_text or len(file_chunks) == 0:
            failed_files.append(
                (uploaded_file.name, "No extractable text was found.")
            )
            continue

        all_new_chunks.extend(file_chunks)
        all_new_metadata.extend(file_metadata)
        all_new_ids.extend(file_ids)
        newly_indexed_files.append(uploaded_file.name)

    embedding_dimension = None

    if all_new_chunks:
        progress.progress(65, text="Creating embeddings for new PDF chunks...")

        embedding_start = time.perf_counter()
        chunk_embeddings = create_embeddings(all_new_chunks)
        embedding_time = time.perf_counter() - embedding_start

        embedding_dimension = int(chunk_embeddings.shape[1])

        progress.progress(85, text="Saving chunks and embeddings in ChromaDB...")

        store_chunks(
            ids=all_new_ids,
            chunks=all_new_chunks,
            embeddings=chunk_embeddings,
            metadatas=all_new_metadata,
        )

    preprocessing_total_time = time.perf_counter() - preprocessing_start_time
    progress.progress(100, text="Persistent indexing completed.")

    return {
        "newly_indexed_files": newly_indexed_files,
        "already_indexed_files": already_indexed_files,
        "failed_files": failed_files,
        "new_chunks": all_new_chunks,
        "new_metadata": all_new_metadata,
        "new_chunk_count": len(all_new_chunks),
        "total_characters": total_characters,
        "text_preview": all_text_preview,
        "embedding_dimension": embedding_dimension,
        "extraction_time": extraction_time,
        "cleaning_chunking_time": cleaning_chunking_time,
        "embedding_time": embedding_time,
        "preprocessing_total_time": preprocessing_total_time,
        "database_chunk_count": count_records(),
    }


def run():
    st.set_page_config(
        page_title="RAG PDF Assistant",
        page_icon="📄",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "level3_indexing_stats" not in st.session_state:
        st.session_state.level3_indexing_stats = None

    with st.sidebar:
        st.title("RAG Controls")

        st.markdown("### App Info")
        st.info("Level 3 — Persistent Multi-PDF RAG")

        st.markdown("### Current Upgrade")
        st.markdown(
            """
            - All Level 2 features retained
            - Persistent ChromaDB vector store
            - Duplicate indexing prevention
            - Search works after app restart
            """
        )

        st.markdown("---")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Built with Streamlit · ChromaDB · Gemini · RAG")

    st.title("RAG PDF Assistant — Level 3")
    st.caption(
        "Level 2 multi-PDF conversational RAG upgraded with persistent ChromaDB storage."
    )

    st.markdown("---")
    st.subheader("Upload and Index Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.expander("📁 Uploaded Files", expanded=True):
            for uploaded_file in uploaded_files:
                st.write(f"✅ {uploaded_file.name}")

    st.markdown("### Retrieval & Chunking Settings")

    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=3,
        )

    with control_col2:
        chunk_size = st.slider(
            "Chunk Size",
            min_value=300,
            max_value=1500,
            value=800,
            step=100,
        )

    with control_col3:
        overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=400,
            value=150,
            step=50,
        )

    if overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
        st.stop()

    if uploaded_files:
        if st.button("Index PDFs in ChromaDB", type="primary"):
            try:
                stats = _index_uploaded_files(
                    uploaded_files=uploaded_files,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                st.session_state.level3_indexing_stats = stats
            except Exception as error:
                st.error("Persistent indexing failed.")
                st.write(f"Technical details: {error}")
                st.stop()
    else:
        st.info(
            "Upload PDFs only when adding new documents. "
            "Already indexed PDFs remain searchable below."
        )

    stats = st.session_state.level3_indexing_stats

    if stats is not None:
        if stats["newly_indexed_files"]:
            st.success(
                "Newly indexed: "
                + ", ".join(stats["newly_indexed_files"])
            )

        if stats["already_indexed_files"]:
            st.info(
                "Already indexed with the same settings: "
                + ", ".join(stats["already_indexed_files"])
            )

        for file_name, reason in stats["failed_files"]:
            st.warning(f"{file_name}: {reason}")

        st.markdown("---")
        st.subheader("Document Processing Summary")

        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

        with metric_col1:
            st.metric("New Characters", stats["total_characters"])

        with metric_col2:
            st.metric("New Chunks", stats["new_chunk_count"])

        with metric_col3:
            st.metric(
                "Embedding Dim",
                stats["embedding_dimension"]
                if stats["embedding_dimension"] is not None
                else "Reused",
            )

        with metric_col4:
            st.metric("Chunk Size", chunk_size)

        with metric_col5:
            st.metric("DB Chunks", stats["database_chunk_count"])

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

        with perf_col1:
            st.metric("Extraction", f"{stats['extraction_time']:.2f}s")

        with perf_col2:
            st.metric(
                "Clean + Chunk",
                f"{stats['cleaning_chunking_time']:.2f}s",
            )

        with perf_col3:
            st.metric("Embedding", f"{stats['embedding_time']:.2f}s")

        with perf_col4:
            st.metric(
                "Total Prep",
                f"{stats['preprocessing_total_time']:.2f}s",
            )

        if stats["text_preview"]:
            with st.expander("Extracted Text Preview"):
                st.write(stats["text_preview"])

        if stats["new_chunks"]:
            with st.expander("New Chunk Preview & Metadata"):
                selected_chunk_number = st.number_input(
                    "Preview new chunk number",
                    min_value=1,
                    max_value=len(stats["new_chunks"]),
                    value=1,
                )

                selected_index = selected_chunk_number - 1
                selected_metadata = stats["new_metadata"][selected_index]

                st.write(f"Source PDF: {selected_metadata['source']}")
                st.write(f"Page number: {selected_metadata['page']}")
                st.write(f"Chunk ID: {selected_metadata['chunk_id']}")
                st.write(stats["new_chunks"][selected_index])

    st.markdown("---")
    st.subheader("Search Scope")

    scope_options = _build_scope_options()

    selected_scope_label = st.selectbox(
        "Search inside",
        list(scope_options.keys()),
    )

    where_filter = scope_options[selected_scope_label]
    chunks_in_scope = count_records(where_filter)

    scope_col1, scope_col2 = st.columns(2)

    with scope_col1:
        st.info(f"Searching in: {selected_scope_label}")

    with scope_col2:
        st.metric("Chunks in Scope", chunks_in_scope)

    if chunks_in_scope == 0:
        st.warning("No persistent chunks exist yet. Upload and index PDFs first.")
        return

    st.markdown("---")
    st.subheader("💬 Ask the Documents")

    question = st.text_input(
        "Ask a question from the PDFs",
        placeholder="Example: What is the main objective?",
    )

    st.subheader("🗨️ Conversation")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question.strip() == "":
        st.warning("Enter a question to generate an answer.")
        return

    if st.button("Generate Answer", type="primary"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        recent_history = st.session_state.messages[-5:-1]

        history_text = ""
        for message in recent_history:
            history_text += (
                f"{message['role']}: {message['content']}\n"
            )

        memory_enhanced_question = f"""
Previous conversation:
{history_text}

Current question:
{question}
"""

        effective_top_k = min(top_k, chunks_in_scope)

        try:
            retrieval_start_time = time.perf_counter()

            query_embedding = create_query_embedding(
                memory_enhanced_question
            )

            retrieved_chunks = search_chroma(
                query_embedding=query_embedding,
                top_k=effective_top_k,
                where_filter=where_filter,
            )

            retrieval_time = time.perf_counter() - retrieval_start_time
        except Exception as error:
            st.error("Retrieval from ChromaDB failed.")
            st.write(f"Technical details: {error}")
            st.stop()

        if len(retrieved_chunks) == 0:
            st.warning("No relevant chunks were retrieved.")
            st.stop()

        llm_start_time = time.perf_counter()
        answer = generate_answer(
            memory_enhanced_question,
            retrieved_chunks,
        )
        llm_time = time.perf_counter() - llm_start_time
        answer_total_time = retrieval_time + llm_time

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Final Answer", "Sources", "Pipeline", "Performance"]
        )

        with tab1:
            st.subheader("Final Answer")
            st.markdown(answer)

            st.markdown("### Sources Used")

            used_sources = {
                (
                    result["metadata"].get("source", "Unknown PDF"),
                    result["metadata"].get("page", "Unknown page"),
                )
                for result in retrieved_chunks
            }

            for source, page in sorted(used_sources):
                st.write(f"- {source}, Page {page}")

        with tab2:
            st.subheader("Retrieved Source Chunks")

            for rank, result in enumerate(retrieved_chunks, start=1):
                metadata = result["metadata"]

                with st.expander(f"Source Chunk {rank}"):
                    st.write(
                        f"Source PDF: {metadata.get('source', 'Unknown PDF')}"
                    )
                    st.write(
                        f"Page number: {metadata.get('page', 'Unknown page')}"
                    )
                    st.write(
                        f"Chunk ID: {metadata.get('chunk_id', 'Unknown')}"
                    )
                    st.write(
                        f"Similarity score: "
                        f"{result['similarity_score']:.4f}"
                    )
                    st.write(result["chunk_text"])

        with tab3:
            st.subheader("Pipeline Insights")

            st.markdown(
                """
                1. Multiple PDFs uploaded  
                2. PDF content hashed for stable identity  
                3. Existing persistent index checked before processing  
                4. Text extracted and cleaned page-by-page  
                5. Text split into overlapping chunks  
                6. Existing embedding pipeline generated vectors  
                7. Chunks, embeddings, and metadata persisted in ChromaDB  
                8. Query embedded using the same embedding model  
                9. ChromaDB performed cosine vector retrieval  
                10. Gemini generated a grounded answer  
                """
            )

            st.write("Question:", question)
            st.write("Search scope:", selected_scope_label)
            st.write("Chunks in scope:", chunks_in_scope)
            st.write("Top-K requested:", top_k)
            st.write("Effective Top-K:", effective_top_k)

        with tab4:
            st.subheader("⏱️ Performance Metrics")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Retrieval Time", f"{retrieval_time:.2f}s")

            with metric_col2:
                st.metric("LLM Generation", f"{llm_time:.2f}s")

            with metric_col3:
                st.metric(
                    "Answer Pipeline",
                    f"{answer_total_time:.2f}s",
                )


if __name__ == "__main__":
    run()
