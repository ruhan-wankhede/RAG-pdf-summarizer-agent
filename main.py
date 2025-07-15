from vectordb import ChromaVectorStore
from mistralai import Mistral
from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from workflow import Workflow
from state import RAGState

load_dotenv()

# first page in the ui allowing for upload
def show_upload_view():
    st.header("AI Document Summarizer")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.session_state.pdf_path = tmp_path
        st.session_state.rag_workflow = Workflow(tmp_path) #initiliaze the langgraph workflow
        st.session_state.app_stage = "summarize" #move to next view
        st.rerun()  # trigger rerun to switch views

    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 0.9rem;
            padding: 10px;
        }
        </style>
        <div class="footer">
            🔗 <a href="https://github.com/ruhan-wankhede/RAG-pdf-summarizer-agent" target="_blank">View this project on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

#actual chat page of the ui
def show_summarize_view():
    st.title("📝 Ask about your document")

    # Rebuild store if needed
    if "rag_workflow" not in st.session_state:
        st.error("No document loaded. Please go back and upload one.")
        return

    # left sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        if st.button("🔙 Upload a Different PDF"):
            st.session_state.clear()
            st.session_state.app_stage = "upload"
            st.rerun()

        st.markdown("## Chat")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_state = RAGState()
            st.rerun()

        st.markdown("---")
        st.markdown(
            """🔗<a href="https://github.com/ruhan-wankhede/RAG-pdf-summarizer-agent" style="text-decoration:none">GitHub Repo</a>""",
            unsafe_allow_html=True
        )

    query = st.chat_input("Ask something about your document...")

    # Initialize chat state if it doesn't exist
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = RAGState()

    # Display chat history
    for message in st.session_state.chat_state.messages:
        if message.type == "human":
            with st.chat_message("user"):
                st.markdown(message.content)
        elif message.type == "ai":
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # get agent response if user enters a chat message
    if query:
        wflow = st.session_state.rag_workflow
        state = RAGState(user_query=query, messages=getattr(st.session_state.get("chat_state", RAGState()), "messages", []))

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                final_state = wflow.run_workflow(state)
                reply = final_state.response
                st.markdown(reply)

        # Store full updated chat state
        st.session_state.chat_state = final_state



if __name__ == "__main__":
    if "app_stage" not in st.session_state:
        st.session_state.app_stage = "upload"
    if st.session_state.app_stage == "upload":
        show_upload_view()
    elif st.session_state.app_stage == "summarize":
        show_summarize_view()

