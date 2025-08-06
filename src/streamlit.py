# src/streamlit_app.py

import streamlit as st
import sys
import os
# --- Now that this file is in 'src', we can import sibling modules directly ---
try:
    from generator import LearningPathGenerator
except ImportError as e:
    st.error(f"""
        **Failed to import LearningPathGenerator.**

        This can happen if the required modules are not in the 'src' directory
        or if there is an issue with the Python environment.

        Please ensure your project structure is correct:
        
        learning-path-agent/
        └── src/
            ├── streamlit_app.py  (this file)
            ├── generator.py
            └── retriever.py

        Error details: {e}
    """)
    st.stop()


# --- Page Configuration ---
# Sets the title, icon, and layout for the browser tab and the app itself.
st.set_page_config(
    page_title="AI Learning Path Generator",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Caching the Generator ---
@st.cache_resource
def load_generator():
    """
    Loads the LearningPathGenerator and caches it using Streamlit's resource caching.
    This is essential for performance, as it prevents the large AI model from being
    reloaded every time the user interacts with a widget.
    """
    try:
        # This single line initializes both the LLM and the retriever.
        return LearningPathGenerator()
    except Exception as e:
        st.error(f"Failed to load the AI model. Please check your setup, dependencies, and model files. Error: {e}")
        st.stop()

# --- Main Application UI ---
st.title("🎓 Personalized Learning-Path Generator")

st.markdown("""
Welcome! This AI agent crafts a personalized, week-by-week learning plan to help you achieve your career goals. 
Just enter your desired career and your current skills below.
""")

# --- User Inputs ---
# Using a form prevents the app from re-running on every keystroke in the input fields.
with st.form("learning_path_form"):
    goal = st.text_input(
        "**Enter your career goal:**",
        placeholder="e.g., Become a Machine Learning Engineer"
    )
    skills = st.text_area(
        "**List your current skills (comma-separated):**",
        placeholder="e.g., Python, SQL, Basic Statistics"
    )
    
    # The button that submits the form's data.
    submitted = st.form_submit_button("✨ Generate My Plan")


# --- Generation Logic ---
# This block only runs when the "Generate My Plan" button is clicked.
if submitted:
    if not goal:
        st.error("Please enter a career goal to generate a plan.")
    else:
        # Load the generator (will be fast after the first run due to caching).
        with st.spinner("Loading AI model... This might take a moment on first launch."):
            generator = load_generator()

        # Generate the plan and show a spinner for user feedback.
        with st.spinner("🤖 The AI is thinking... Crafting your personalized plan..."):
            try:
                learning_plan = generator.generate(goal, skills)
                
                st.success("Your learning plan is ready!")
                st.markdown("---")
                st.markdown(learning_plan)
                st.balloons()
                
            except Exception as e:
                st.error(f"An error occurred during plan generation: {e}")
st.markdown("""
---
*Powered by a locally-run AI model and a FAISS-based retrieval system.*
""")