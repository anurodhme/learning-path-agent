import streamlit as st
import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import CourseRetriever
from src.generator import LearningPathGenerator

# Page configuration
st.set_page_config(
    page_title="Learning Path Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .title-text {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .subtitle-text {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
    }
    .learning-plan-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .course-item {
        background: #f8f9fa;
        border-left: 4px solid #4ECDC4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="title-text">🎓 Learning Path Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Your AI-powered personalized learning companion</p>', unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize the CourseRetriever and LearningPathGenerator."""
    try:
        retriever = CourseRetriever()
        generator = LearningPathGenerator()
        return retriever, generator
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

# Main content area
def main():
    retriever, generator = initialize_components()
    
    if not retriever or not generator:
        st.error("Failed to initialize learning path components. Please check the logs.")
        return

    # Create two columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 What's your learning goal?")
        user_goal = st.text_area(
            "Describe what you want to learn:",
            placeholder="e.g., I want to become a data scientist, learn web development, or master machine learning...",
            height=100,
            key="user_goal"
        )
        
        st.markdown("### 🛠️ Current skills & experience")
        current_skills = st.text_area(
            "What skills do you already have?",
            placeholder="e.g., Basic Python, Excel, Statistics, or any relevant experience...",
            height=80,
            key="current_skills"
        )
    
    with col2:
        st.markdown("### ⚙️ Advanced Settings")
        
        # Model settings
        model_name = st.selectbox(
            "Language Model:",
            ["Qwen/Qwen3-0.6B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            index=0
        )
        
        max_tokens = st.slider("Max Response Length", 100, 1000, 500)
        
        # Search settings
        top_k = st.slider("Number of courses to consider", 5, 20, 10)
        
        # Generation settings
        temperature = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.7)
    
    # Generate button
    if st.button("🚀 Generate My Learning Path", use_container_width=True):
        if not user_goal.strip():
            st.warning("Please enter your learning goal first!")
            return
            
        with st.spinner("🔍 Searching for relevant courses and generating your personalized learning path..."):
            try:
                # Generate learning path
                learning_plan = generator.generate_learning_path(
                    user_goal=user_goal,
                    current_skills=current_skills,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Display results
                st.markdown("### 📋 Your Personalized Learning Path")
                
                # Create a beautiful card for the learning plan
                st.markdown('<div class="learning-plan-card">', unsafe_allow_html=True)
                st.markdown(learning_plan.replace('\n', '<br>'), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download section
                st.markdown("### 💾 Save Your Plan")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📄 Download as Text",
                        data=learning_plan,
                        file_name="learning_path.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create markdown version
                    markdown_content = f"""# My Learning Path

**Goal:** {user_goal}

**Current Skills:** {current_skills}

## Learning Plan
{learning_plan}
"""
                    st.download_button(
                        label="📝 Download as Markdown",
                        data=markdown_content,
                        file_name="learning_path.md",
                        mime="text/markdown"
                    )
                
                with col3:
                    # Create JSON version
                    import json
                    json_data = {
                        "goal": user_goal,
                        "current_skills": current_skills,
                        "learning_plan": learning_plan,
                        "generated_at": str(pd.Timestamp.now())
                    }
                    st.download_button(
                        label="📊 Download as JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name="learning_path.json",
                        mime="application/json"
                    )
                
                # Show relevant courses
                st.markdown("### 📚 Relevant Courses Found")
                try:
                    # Get search results for display
                    search_query = f"{user_goal} {current_skills}"
                    search_results = retriever.search(search_query, top_k=top_k)
                    
                    for i, course in enumerate(search_results, 1):
                        st.markdown(f'<div class="course-item"><strong>{i}. {course}</strong></div>', 
                                   unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying courses: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error generating learning path: {str(e)}")
                st.info("Please try again with a different goal or check the logs for details.")

# Footer
st.markdown("""
    <hr style='margin-top: 3rem;'>
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Made with ❤️ by Learning Path Agent | Powered by AI & Open Source</p>
        <p style='font-size: 0.9rem;'>Built with Streamlit, Sentence Transformers, FAISS, and Hugging Face Transformers</p>
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()