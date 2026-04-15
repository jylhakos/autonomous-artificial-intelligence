"""
Multi-Model Chat Application using LiteLLM

This example demonstrates how to use LiteLLM to interact with multiple
language models through a unified interface.

Requirements:
- streamlit
- litellm
- API keys configured in environment variables
"""

import streamlit as st
from litellm import completion
import os


def get_model_response(model_name: str, prompt: str) -> str:
    """
    Get response from specified language model
    
    Args:
        model_name: Name of the model to use
        prompt: User's input prompt
        
    Returns:
        Model's response as string
    """
    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Main Streamlit application"""
    
    st.title("🤖 Multi-Model Chat")
    st.markdown("Chat with different AI models using a unified interface")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_option = st.selectbox(
            "Choose a language model:",
            (
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4o",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "ollama/llama2",
                "gemini/gemini-pro"
            ),
            help="Select the AI model to use for responses"
        )
        
        # Temperature setting
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random"
        )
        
        # Max tokens
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum length of the response"
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state['chat_history'] = []
            st.rerun()
        
        # Display API key status
        st.markdown("---")
        st.markdown("**API Keys Status:**")
        if os.getenv("OPENAI_API_KEY"):
            st.success("✓ OpenAI")
        if os.getenv("ANTHROPIC_API_KEY"):
            st.success("✓ Anthropic")
        if os.getenv("GOOGLE_API_KEY"):
            st.success("✓ Google")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display chat history
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state['chat_history'].append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get model response
        with st.chat_message("assistant"):
            with st.spinner(f"💭 {model_option} is thinking..."):
                try:
                    response = completion(
                        model=model_option,
                        messages=[
                            {"role": msg['role'], "content": msg['content']}
                            for msg in st.session_state['chat_history']
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    assistant_message = response.choices[0].message.content
                except Exception as e:
                    assistant_message = f"❌ Error: {str(e)}\n\nPlease check your API keys and model availability."
                
                st.write(assistant_message)
        
        # Add assistant message to chat history
        st.session_state['chat_history'].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Rerun to update chat display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Powered by LiteLLM | Supports 100+ LLM providers
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
