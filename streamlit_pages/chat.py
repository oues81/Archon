try:
    from langgraph.types import Command
except ImportError:
    class Command:
        """Dummy implementation for compatibility"""
        pass

import streamlit as st
import uuid
import sys
import os
import asyncio
import logging
import httpx

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_thread_id():
    """Get a unique thread_id for the session."""
    return str(uuid.uuid4())

async def run_agent_with_streaming(user_input: str, thread_id: str):
    """Run the agent by calling the backend API and stream the output chunks."""
    api_url = "http://localhost:8110/invoke"
    payload = {
        "message": user_input,
        "thread_id": thread_id,
        "is_first_message": not st.session_state.get("messages")
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            yield response.json()
    except httpx.RequestError as e:
        logger.error(f"Error communicating with the backend: {e}", exc_info=True)
        yield {"error": f"Could not connect to the agent service at {api_url}. Is it running?"}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        yield {"error": str(e)}

async def chat_tab():
    """Display the chat interface for Archon."""
    st.write("Describe an AI agent you want to build, and I'll code it for you.")
    st.write("Example: Build an agent that can get the weather for a list of locations.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Get a unique thread_id for the session
    thread_id = get_thread_id()

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if user_input := st.chat_input("What do you want to build today?"):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            final_code = ""

            with st.spinner("Archon is thinking..."):
                try:
                    # Call the agent and get the response
                    response_data = None
                    async for data in run_agent_with_streaming(user_input, thread_id):
                        response_data = data

                    if response_data and not response_data.get("error"):
                        # Assuming the response contains the generated code in a 'response' key
                        # based on API_REFERENCE.md
                        final_code = response_data.get("response", "")
                        if final_code:
                            full_response = f"\n```python\n{final_code}\n```\n"
                            response_placeholder.markdown(full_response)
                        else:
                            full_response = "\nSorry, I couldn't generate the code. Please try again.\n"
                            response_placeholder.markdown(full_response)
                    else:
                        error_message = response_data.get("error", "An unknown error occurred.")
                        full_response = f"An error occurred: {error_message}"
                        response_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"An error occurred: {e}"
                    logger.error(f"Error during streaming: {e}", exc_info=True)
                    response_placeholder.markdown(full_response)
            
            # Add the final assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})