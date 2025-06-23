import streamlit as st
import base64
from langchain_ollama.llms import OllamaLLM

# Configure page
st.set_page_config(
    page_title="Ollama Vision Analysis",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ollama_model(model_name="llava"):
    """Load and cache the Ollama model to avoid reloading on every request"""
    try:
        return OllamaLLM(model=model_name)
    except Exception as e:
        st.error(f"Failed to load Ollama model '{model_name}': {str(e)}")
        return None

def encode_image_to_base64(uploaded_file):
    """Convert uploaded image to base64 string"""
    try:
        uploaded_file.seek(0)  # Reset file pointer
        return base64.b64encode(uploaded_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def analyze_image_with_ollama(llm, image_base64, prompt):
    """Analyze image using Ollama vision model"""
    if not llm:
        return "❌ Ollama model not available. Please check your Ollama installation."
    
    try:
        # For Ollama vision models, pass the image directly in the images parameter
        response = llm.invoke(prompt, images=[image_base64])
        return response
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"

# Initialize session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Main title and description
st.title("🖼️ Ollama Vision Analysis")
st.markdown("Upload a JPG image and ask questions about it using Ollama's vision capabilities!")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection - Only vision models
    model_options = ["llava", "llava:7b", "llava:13b", "llava:34b", "bakllava", "gemma3:4b"]
    selected_model = st.selectbox(
        "Choose Vision Model",
        options=model_options,
        index=0,
        help="Select the Ollama vision model to use for analysis"
    )
    
    # Load model
    llm = load_ollama_model(selected_model)
    
    if llm:
        st.success(f"✅ {selected_model} model loaded")
    else:
        st.error(f"❌ Failed to load {selected_model}")
        st.info("Make sure Ollama is running and the model is installed")
    
    # Clear history button
    if st.button("🗑️ Clear History", help="Clear analysis history"):
        st.session_state.analysis_history = []
        st.success("History cleared!")
    
    # Model info
    with st.expander("📚 Model Information"):
        st.markdown(f"""
        **Current Model:** {selected_model}
        
        **Available Vision Models:**
        - `llava`: General vision model (7B)
        - `llava:7b`: 7B parameter version
        - `llava:13b`: 13B parameter version (more accurate)
        - `llava:34b`: 34B parameter version (most accurate)
        - `bakllava`: Alternative vision model
        - `moondream`: Lightweight vision model
        
        To install a model:
        ```bash
        ollama pull {selected_model}
        ```
        
        **Note:** Only vision-capable models are listed.
        """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload & Configure")
    
    # File uploader with improved configuration
    uploaded_file = st.file_uploader(
        "Choose a JPG image", 
        type=['jpg', 'jpeg', 'png'],  # Added PNG support
        help="Upload a JPG, JPEG, or PNG image file for analysis",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size:,} bytes",
            "File type": uploaded_file.type
        }
        st.json(file_details)
        
        # Display the uploaded image - Fix deprecation warning
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Text input for the question/prompt
        user_prompt = st.text_area(
            "What would you like to know about this image?",
            value="Describe what you see in this image in detail.",
            height=100,
            help="Enter your question or request about the image"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if not user_prompt.strip():
                st.warning("⚠️ Please enter a question or prompt about the image.")
            elif not llm:
                st.error("❌ Ollama model is not available. Please check your installation.")
            else:
                with st.status("Analyzing image with Ollama...", expanded=True) as status:
                    st.write("🔄 Converting image to base64...")
                    image_base64 = encode_image_to_base64(uploaded_file)
                    
                    if image_base64:
                        st.write(f"🤖 Getting analysis from {selected_model}...")
                        result = analyze_image_with_ollama(llm, image_base64, user_prompt)
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            "image_name": uploaded_file.name,
                            "prompt": user_prompt,
                            "result": result,
                            "model": selected_model
                        })
                        
                        status.update(label="✅ Analysis complete!", state="complete")
                    else:
                        status.update(label="❌ Failed to process image", state="error")

with col2:
    st.header("📝 Results")
    
    if st.session_state.analysis_history:
        # Show latest result prominently
        latest = st.session_state.analysis_history[-1]
        
        st.subheader("🔍 Latest Analysis")
        with st.container():
            st.markdown(f"**Image:** {latest['image_name']}")
            st.markdown(f"**Model:** {latest.get('model', 'Unknown')}")
            st.markdown(f"**Question:** {latest['prompt']}")
            
            # Display result with better formatting
            if latest['result'].startswith('❌'):
                st.error(latest['result'])
            else:
                st.success("Analysis completed successfully!")
                st.markdown("**Answer:**")
                st.write(latest['result'])
        
        # Show history if there are multiple analyses
        if len(st.session_state.analysis_history) > 1:
            st.subheader("📚 Analysis History")
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[:-1]), 1):
                with st.expander(f"Analysis {len(st.session_state.analysis_history) - i}: {analysis['image_name']}"):
                    st.markdown(f"**Model:** {analysis.get('model', 'Unknown')}")
                    st.markdown(f"**Question:** {analysis['prompt']}")
                    st.markdown(f"**Answer:** {analysis['result']}")
    else:
        st.info("👆 Upload an image and analyze it to see results here.")

# Instructions and help
if not uploaded_file:
    st.info("👆 Please upload an image to get started.")
    
    # Show helpful information when no image is uploaded
    with st.expander("ℹ️ How to use this app", expanded=True):
        st.markdown("""
        ### 🚀 Getting Started
        1. **Upload an image**: Use the file uploader on the left to select a JPG, JPEG, or PNG image
        2. **Choose a model**: Select your preferred Ollama vision model from the sidebar
        3. **Ask a question**: Enter what you'd like to know about the image
        4. **Get analysis**: Click the "Analyze Image" button to get Ollama's response
        
        ### 📋 Example Questions
        - "Describe what you see in this image in detail"
        - "What objects are visible in this image?"
        - "What is the main subject of this image?"
        - "Describe the colors and composition"
        - "What text can you read in this image?"
        
        ### ⚡ Requirements
        Make sure you have Ollama installed and a vision model downloaded:
        ```bash
        # Install Ollama (if not already installed)
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Start Ollama service
        ollama serve
        
        # Pull a vision model
        ollama pull llava
        ```
        
        ### 🎯 Tips
        - Use specific questions for better results
        - Try different models for varying response styles
        - Check the sidebar for model information and settings
        - Make sure Ollama is running in the background
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit �� and Ollama 🦙")

