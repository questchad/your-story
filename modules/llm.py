import base64
import re
import platform
import sys
from time import perf_counter, sleep, time

from langchain.agents import create_agent
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_agent(config):
    logger.info("Load agent: Start")
    llm = init_chat_model(
        model=config["LLM_ID"],
        model_provider=config["LLM_PROVIDER"],
        temperature=config["TEMPERATURE"]    
    )

    # Automatic prompt caching
    # Will rarely be invoked due to token minimums https://platform.claude.com/docs/en/build-with-claude/prompt-caching#cache-limitations
    middleware = [AnthropicPromptCachingMiddleware()] if config["LLM_PROVIDER"] == "anthropic" else None
    agent = create_agent(
        llm,
        middleware= middleware,
        checkpointer=InMemorySaver(), # Short term memory
    )
    
    # For Ollama, dummy invocation to preload the local model
    if config["LLM_PROVIDER"]=="ollama":
        logger.info(f"Preloading {id} on Ollama")
        response = agent.invoke(
            input=create_input(agent_instruction="hello"),
            config={"configurable": {"thread_id": str(int(time()))}}, # Use a unique thread ID, so it doesn't get added to message history nor conflict with another user using app
        )
    logger.info("Load agent: Done")
    return agent


def create_input(agent_instruction, user_message=None, add_image=False, system_role=None):
    messages = []

    if st.session_state.get("stage", None) == "chat_welcome":
        # Add system role to the first message
        messages.append({"role": "system", "content": system_role})

    if user_message:
        messages.append(
            {"role": "user", "content": user_message}
        )
    
    # Add agent instruction
    # With or without image
    if "image_for_llm" in st.session_state and agent_instruction and add_image:
        image_base64_bytes = st.session_state["image_for_llm"]
        image_base64_str = base64.b64encode(image_base64_bytes).decode("utf-8")
        messages.append(
            # Works for both Anthropic (as long as we decode the bytes into string first ^) and Ollama (can take bytes or string input, apprarently auto-converts before invocation)
            {"role": "user", "content": [
                    {"type": "text", "text": agent_instruction}, 
                    {"type": "image", "base64": image_base64_str, "mime_type": "image/jpeg"}
                ]
            }
            )
    else:
        messages.append(
            {"role": "user", "content": agent_instruction}
        )

    return {"messages": messages}


def clean_message_for_log(message_text):
    """Remove giant image data from LLM message history when logging, to make logs easier to read
    
    Does not change the data going to LLM or stored in conversation history
    """

    pattern = r"(?<=base64).*?(?=mime_type)"
    match = re.search(pattern, message_text, re.DOTALL)
    if match:
        image_data = match.group(0).strip()
        return message_text.replace(image_data, f"': **Image data {len(image_data)} chars**, '")
    return message_text
    

def invoke_agent(agent, input, debug):
    if platform.system == "Linux":
        sys.stdout.flush()
    llm_start = perf_counter()
    if debug:
        with st.spinner("LLM processing"):
            llm_response = agent.invoke(
                input = input,
                config = {"configurable": {"thread_id": st.session_state["memory_thread_id"]}},
            )
    else:
        llm_response = agent.invoke(
            input = input,
            config = {"configurable": {"thread_id": st.session_state["memory_thread_id"]}},
        )

    agent_message = llm_response["messages"][-1].content  # Should always be an AIMessage. TODO: Validate?
    llm_end = perf_counter()
    logger.info(f"LLM: {llm_end-llm_start} seconds")
    
    logger.info(f"""LLM, All messages: 
        {[clean_message_for_log(str(msg.content)) for msg in llm_response["messages"]]}
    """)
    logger.info(f"LLM, Usage: {llm_response["messages"][-1].usage_metadata}") 

    if platform.system == "Linux":
        sys.stdout.flush()

    return agent_message

