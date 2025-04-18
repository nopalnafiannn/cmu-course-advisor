"""
Streamlit UI for the CMU Course Advisor Chatbot with user profiling and RAG support.
Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st
from PIL import Image
import os

# Monkey patch to fix the torch classes issue with Streamlit file watcher
import sys
import monkeypatch  # This will be imported but not used directly

# Set page configuration once at startup
st.set_page_config(
    page_title="CMU Course Advisor", 
    page_icon=":mortar_board:",
    layout="wide"
)

# Define CMU colors
CMU_RED = "#C41230"
CMU_GRAY = "#63666A"
CMU_GOLD = "#C6AC8F"
CMU_BLACK = "#000000"
CMU_WHITE = "#FFFFFF"

# Custom CSS for styling
st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    h1, h2, h3 {{
        color: {CMU_RED};
    }}
    .stButton button {{
        background-color: {CMU_RED};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }}
    .stButton button:hover {{
        background-color: #9E0E26;
    }}
    .chat-message-assistant {{
        background-color: {CMU_GOLD}22;
        border-left: 5px solid {CMU_RED};
        padding: 10px;
    }}
    .stTextInput, .stSelectbox {{
        border-color: {CMU_RED};
    }}
    .stForm {{
        border: 1px solid {CMU_GRAY};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    footer {{
        visibility: hidden;
    }}
    .student-info {{
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding: 10px;
        background-color: {CMU_RED}11;
        border-radius: 5px;
    }}
    </style>
""", unsafe_allow_html=True)

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

# Create header with logos and course information
def create_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Display CMU logo
        logo_path = os.path.join(os.path.dirname(__file__), "images", "cmu_logo.png")
        if os.path.exists(logo_path):
            cmu_logo = Image.open(logo_path)
            st.image(cmu_logo, width=200)
    
    with col2:
        # Main title and course info
        st.title("CMU Course Advisor")
        st.markdown(f"<h2 style='color:{CMU_RED}; font-size:1.5em;'>Final Project: Introduction to AI (95-891)</h2>", unsafe_allow_html=True)
        
        # Student information
        st.markdown(f"""
            <div class='student-info'>
                <b>Students:</b> Naufal Nafian, Pablo Zavala
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Display Heinz Hall picture
        heinz_path = os.path.join(os.path.dirname(__file__), "images", "heinz_hall_picture.jpg")
        if os.path.exists(heinz_path):
            heinz_pic = Image.open(heinz_path)
            st.image(heinz_pic, width=200)
    
    # Horizontal separator
    st.markdown(f"<hr style='height:2px;border-width:0;color:{CMU_RED};background-color:{CMU_RED}'>", unsafe_allow_html=True)

# Display the header
create_header()

# Initialization with spinner
with st.spinner("Initializing document index and RAG pipeline..."):
    # Load or retrieve cached document store and RAG pipeline
    document_store, pipeline = init()

# Initialize session state for user profile and chat history
if 'profile' not in st.session_state:
    st.subheader("🎓 Welcome to the CMU Course Advisor!")
    st.markdown("""
        <div style='background-color:#f0f0f0; padding:15px; border-radius:5px; border-left:5px solid #C41230; color:#000000;'>
            This AI-powered advisor helps you find the perfect courses for your academic journey at Carnegie Mellon University. Let's start by getting to know you.
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("profile_form", border=False):
        st.markdown(f"<h3 style='color:{CMU_RED};'>Your Profile</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            interest = st.text_input("What is your interest field?", placeholder="e.g., Data Science, Policy, Management...")
        with col2:
            level = st.selectbox("What is your current level?", ["Beginner", "Intermediate", "Expert"])
        
        submitted = st.form_submit_button("Start Chatting", use_container_width=True)
        if submitted:
            # Save profile and reset history
            st.session_state.profile = f"Interest field: {interest}. Current level: {level.lower()}."
            st.session_state.history = []
            # Add a welcome message from the assistant
            welcome_msg = f"Hello! I'm your CMU course advisor. I see you're interested in {interest}. How can I help you find courses today? Are you looking for recommendations in your field of interest?"
            st.session_state.history.append({"role": "assistant", "content": welcome_msg})
            # Rerun to update the UI with the new profile
            st.rerun()
else:
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat with the Course Advisor")
        
        # User profile display
        st.markdown(f"""
            <div style='background-color:{CMU_RED}22; padding:10px; border-radius:5px; margin-bottom:15px; color:{CMU_RED};'>
                <b>Your Profile:</b> {st.session_state.profile}
            </div>
        """, unsafe_allow_html=True)
        
        # Chat container with custom styling
        chat_container = st.container(height=400, border=False)
        
        with chat_container:
            if 'history' not in st.session_state:
                st.session_state.history = []

            # Display chat history
            for msg in st.session_state.history:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👨‍🎓"):
                        st.markdown(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])

        # Prompt for new question
        user_input = st.chat_input("Ask me about courses...", key="chat_input")
        if user_input:
            # Append user message to history
            st.session_state.history.append({"role": "user", "content": user_input})
            # Display the user message immediately
            with st.chat_message("user", avatar="👨‍🎓"):
                st.markdown(user_input)
            
            # Show a loading spinner while processing
            with st.spinner("Thinking..."):
                try:
                    # Get the answer (sources are included in the UI)
                    answer, sources = answer_query(
                        document_store, pipeline, user_input, st.session_state.profile, st.session_state.history
                    )
                    
                    # Append assistant's response to history (without sources)
                    st.session_state.history.append({"role": "assistant", "content": answer})
                    
                    # Display the assistant's response immediately
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(answer)
                except Exception as e:
                    # Handle any errors that occur during processing
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.history.append({"role": "assistant", "content": error_msg})
                    
                    # Display the error message immediately
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(error_msg)
                    st.error(f"Technical details: {type(e).__name__}: {str(e)}")
    
    with col2:
        # Sidebar with information about the advisor
        st.markdown(f"""
            <div style='background-color:{CMU_GOLD}33; padding:15px; border-radius:5px; color:{CMU_RED};'>
                <h3 style='color:{CMU_RED}; font-size:1.2em;'>About this Advisor</h3>
                <p style='color:{CMU_WHITE};'>This AI course advisor uses advanced RAG (Retrieval-Augmented Generation) technology to provide personalized course recommendations from the Heinz College catalog.</p>
                <p style='color:{CMU_WHITE};'>It can help you:</p>
                <ul style='color:{CMU_WHITE};'>
                    <li>Find courses in specific subject areas</li>
                    <li>Get detailed course information</li>
                    <li>Plan your academic journey</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Option to reset the conversation
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Conversation", use_container_width=True):
            for key in ["profile", "history"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Rerun to reset the UI state
            st.rerun()

# Footer
st.markdown(f"""
    <div style='margin-top:2rem; text-align:center; color:{CMU_GRAY}; font-size:0.8em; background-color:transparent;'>
        <hr style='height:1px;border-width:0;color:{CMU_GRAY};background-color:{CMU_GRAY}'>
        © 2025 Carnegie Mellon University, Heinz College | Introduction to AI (95-891)
    </div>
""", unsafe_allow_html=True)