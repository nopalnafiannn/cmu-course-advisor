"""
Streamlit UI for the CMU Course Advisor Chatbot with user profiling and RAG support.
Run with:
    streamlit run streamlit_app_patched.py
"""
# Monkey patch to fix the torch classes issue with Streamlit file watcher
import sys
import monkeypatch  # This will be imported but not used directly

import streamlit as st
# Set page configuration once at startup
st.set_page_config(page_title="CMU Course Advisor Chatbot", page_icon=":mortar_board:")

# Import from our updated haystack_rag_advisor_profiled module
try:
    from haystack_rag_advisor_profiled import build_index, create_rag_pipeline, answer_query
except ImportError as e:
    st.error(
        f"Import error: {str(e)}. "
        "Please install dependencies via 'pip install -r requirements.txt' "
        "and ensure your virtual environment is activated."
    )
    import sys
    st.exception(sys.exc_info()[1])
    st.stop()

 # Cache the document store and pipeline to avoid rebuilding on every interaction
@st.cache_resource
def init():
    document_store = build_index()
    pipeline = create_rag_pipeline(document_store)
    return document_store, pipeline

with st.spinner("Initializing document index and RAG pipeline..."):
    # Load or retrieve cached document store and RAG pipeline
    document_store, pipeline = init()

# Initialize session state for user profile and chat history
if 'profile' not in st.session_state:
    st.title("CMU Course Advisor Chatbot")
    st.subheader("Welcome! Let's start by getting to know you.")
    with st.form("profile_form"):
        interest = st.text_input("What is your interest field?")
        level = st.selectbox("What is your current level?", ["Beginner", "Intermediate", "Expert"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            # Save profile and reset history
            st.session_state.profile = f"Interest field: {interest}. Current level: {level.lower()}."
            st.session_state.history = []
            # Rerun to update the UI with the new profile
            st.rerun()
else:
    # Main chat interface
    st.title("CMU Course Advisor Chatbot")
    st.markdown(f"**Your Profile:** {st.session_state.profile}")
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

    # Prompt for new question
    user_input = st.chat_input("Ask me about courses...")
    if user_input:
        # Append and display the user's message immediately
        st.session_state.history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Show a loading spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Get the answer (sources are ignored in the UI)
                answer, sources = answer_query(
                    document_store, pipeline, user_input, st.session_state.profile
                )
                
                # Add sources to the answer if available
                if sources:
                    source_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                    answer += source_text
                    
                # Append and display the assistant's response immediately
                st.session_state.history.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
            except Exception as e:
                # Handle any errors that occur during processing
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.history.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").write(error_msg)
                st.error(f"Technical details: {type(e).__name__}: {str(e)}")

    # Option to reset the conversation
    if st.button("Reset Conversation"):
        for key in ["profile", "history"]:
            if key in st.session_state:
                del st.session_state[key]
        # Rerun to reset the UI state
        st.rerun()