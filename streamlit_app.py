import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# ───────────────────────────────────────────────────────────────
# Streamlit page config – MUST be first Streamlit command
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM PDF Explorer", layout="wide")

# Local helper functions
from functions import (
    get_pdf_text,
    create_vectorstore_from_texts,
    query_document,
    requires_project_id,
)

# ───────────────────────────────────────────────────────────────
# Environment & session bootstrap
# ───────────────────────────────────────────────────────────────
load_dotenv()

st.session_state.setdefault("api_key", os.getenv("OPENAI_API_KEY", ""))
st.session_state.setdefault("project_id", os.getenv("OPENAI_PROJECT_ID", ""))
st.session_state.setdefault("vectorstore", None)

# ───────────────────────────────────────────────────────────────
# Sidebar – credentials
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Settings")

    key_input = st.text_input(
        "OpenAI API key",
        value=st.session_state["api_key"],
        type="password",
        placeholder="sk-…  or  sk-proj-…",
    )

    proj_input = st.text_input(
        "OpenAI Project ID (only for sk-proj keys)",
        value=st.session_state["project_id"],
        placeholder="proj_…",
    )

    col_save, col_clear = st.columns(2)
    if col_save.button("Save", use_container_width=True):
        st.session_state["api_key"] = key_input.strip()
        st.session_state["project_id"] = proj_input.strip()
        st.success("Credentials saved.")

    if col_clear.button("Clear", use_container_width=True):
        st.session_state.pop("api_key", None)
        st.session_state.pop("project_id", None)
        st.info("Credentials cleared.")

    st.caption("Keys live only in this browser session – never stored.")

if requires_project_id(st.session_state["api_key"]) and not st.session_state["project_id"]:
    st.warning("Project‑scoped key detected – please supply the matching Project ID.")

# ───────────────────────────────────────────────────────────────
# Helper – inline PDF preview
# ───────────────────────────────────────────────────────────────

def display_pdf(uploaded_file):
    if uploaded_file is None:
        return
    b64 = base64.b64encode(uploaded_file.getvalue()).decode()
    st.markdown(
        f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='800'></iframe>",
        unsafe_allow_html=True,
    )

# ───────────────────────────────────────────────────────────────
# Layout
# ───────────────────────────────────────────────────────────────
left, right = st.columns([0.5, 0.5], gap="large")

with left:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

with right:
    display_pdf(uploaded_file)

# ───────────────────────────────────────────────────────────────
# Vector‑store creation
# ───────────────────────────────────────────────────────────────
if uploaded_file is not None:
    if not st.session_state["api_key"]:
        st.warning("Save a valid OpenAI key first.")
    elif requires_project_id(st.session_state["api_key"]) and not st.session_state["project_id"]:
        st.warning("Project ID required for this key.")
    else:
        with st.spinner("Embedding document…"):
            docs = get_pdf_text(uploaded_file)
            try:
                st.session_state["vectorstore"] = create_vectorstore_from_texts(
                    docs,
                    api_key=st.session_state["api_key"],
                    project_id=st.session_state["project_id"] or None,
                    file_name=uploaded_file.name,
                )
                st.success("PDF indexed. Ask your question below.")
            except Exception as e:
                st.error(f"Embedding step failed: {e}")
                raise

# ───────────────────────────────────────────────────────────────
# Query interface
# ───────────────────────────────────────────────────────────────
if st.session_state["vectorstore"] is not None:
    with left:
        query = st.text_area("Your question:")
        if st.button("Ask") and query.strip():
            with st.spinner("Generating answer…"):
                try:
                    answer = query_document(
                        st.session_state["vectorstore"],
                        query=query,
                        api_key=st.session_state["api_key"],
                        project_id=st.session_state["project_id"] or None,
                    )
                    st.session_state["last_answer"] = answer
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

    if "last_answer" in st.session_state:
        st.subheader("Answer")
        st.write(st.session_state["last_answer"])

