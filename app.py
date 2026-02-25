import streamlit as st
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="OpenRouter Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat interface
st.markdown("""
<style>
    .stChatMessage {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_client(api_key):
    """Initialize the OpenAI client with OpenRouter configuration"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize client: {str(e)}")
        return None

def get_api_key():
    """Get API key from various sources"""
    # Try to get from environment first
    api_key = os.getenv("API_KEY")
    
    # If not in environment, try from session state
    if not api_key and "api_key" in st.session_state:
        api_key = st.session_state.api_key
    
    return api_key

def main():
    # Title and description
    st.title("💬 OpenRouter AI Chat")
    st.markdown("Chat with various AI models through OpenRouter")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "API Key",
            type="password",
            value=get_api_key() if get_api_key() else "",
            help="Enter your OpenRouter API key. It will be stored in session state."
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            [
                "openai/gpt-5.2",
                "openai/gpt-4",
                "openai/gpt-3.5-turbo",
                "anthropic/claude-2",
                "google/palm-2",
                "meta-llama/llama-2-70b"
            ],
            index=0,
            help="Choose the AI model for chat"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_tokens = st.slider(
                "Max Tokens",
                min_value=50,
                max_value=4096,
                value=512,
                step=50,
                help="Maximum number of tokens in the response"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Controls randomness: Lower values are more deterministic"
            )
            
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Nucleus sampling parameter"
            )
        
        # Website configuration
        st.subheader("🌐 Website Info")
        site_url = st.text_input(
            "Site URL",
            value="http://localhost:8501",
            help="Your website URL for OpenRouter headers"
        )
        
        site_name = st.text_input(
            "Site Name",
            value="Streamlit Chat",
            help="Your site name for OpenRouter headers"
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            st.rerun()
        
        # Credits information
        st.info(
            "💡 **Note:** Make sure you have sufficient credits in your OpenRouter account. "
            "Visit [OpenRouter Settings](https://openrouter.ai/settings/credits) to check your credits."
        )
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    
    # Initialize client
    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to start chatting.")
        st.stop()
    
    client = initialize_client(api_key)
    if not client:
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't display system message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with spinner
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Prepare messages for API call (excluding system message if present)
                api_messages = [
                    msg for msg in st.session_state.messages 
                    if msg["role"] != "system"
                ]
                
                # Make API call
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": site_url if site_url else "http://localhost:8501",
                        "X-OpenRouter-Title": site_name if site_name else "Streamlit Chat",
                    },
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    messages=api_messages,
                    stream=True  # Enable streaming for better UX
                )
                
                # Stream the response
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)  # Small delay for visual effect
                
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                
            except openai.APIStatusError as e:
                msg = str(e)
                if "requires more credits" in msg or "can only afford" in msg:
                    st.error(
                        "❌ **Insufficient Credits!**\n\n"
                        "Your OpenRouter account needs more credits. "
                        "[Upgrade your plan](https://openrouter.ai/settings/credits) "
                        "or reduce `max_tokens` in advanced settings."
                    )
                elif "model" in msg.lower() and "not found" in msg.lower():
                    st.error(f"❌ Model '{model}' not available. Please select another model.")
                else:
                    st.error(f"❌ API request failed: {msg}")
            
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()