import os
import platform
import random
import sys
import tempfile
from time import perf_counter, time

from elevenlabs.client import ElevenLabs
import langchain
from langchain.messages import AIMessage
from loguru import logger
import streamlit as st
import streamlit.components.v1 as components

from modules.audio import load_whisper, text_to_speech, play_speech
from modules.image import process_image
from modules.llm import load_agent, create_input, invoke_agent
from modules.utils import load_config

# Get logs to show up on ST Cloud
if platform.system == "Linux":
    sys.stdout.flush()


# Must come before the first st. content call, so since audio_player is a global outside main(), this must come first
st.set_page_config(page_title="Your Journey", page_icon="🤳", initial_sidebar_state="collapsed")

# Create placeholder for player widget, so we can clear as needed
# TODO: Make this not a global variable
audio_player = st.empty()

config = load_config()


@st.cache_resource()
def load_elevenlabs_client():
    return ElevenLabs(
         api_key=os.getenv("ELEVENLABS_API_KEY_DEV"),
    )


def initialize_app():
    logger.info("Initializing app")
    if "stage" not in st.session_state:
        st.session_state["stage"] = "waiting_for_upload"
        st.header("Please upload an image to start")

    agent = load_agent(config)
    elevenlabs_client = load_elevenlabs_client()

    if "questions_asked" not in st.session_state:
        st.session_state["questions_asked"] = 0

    # TODO: Change all these IDs to more semantic "mic_id_1" that we can still increment
    if "uploaded_image_id" not in st.session_state:
        st.session_state["uploaded_image_id"] = 1001
    if "file_uploader_id" not in st.session_state:
        st.session_state["file_uploader_id"] = 10001
    if "memory_thread_id" not in st.session_state:
        st.session_state["memory_thread_id"] = str(int(time())) # Create a new ID unique to this user's interaction
    
    return agent, elevenlabs_client


def restart_app():
    logger.info("Restarting app")
    st.cache_data.clear()
    resettable_keys = [key for key in ["stage", "questions_asked", "image_for_llm", "image_for_display", "user_recording"] if key in st.session_state.keys()]
    for key in resettable_keys:
        del st.session_state[key]   
    st.session_state["file_uploader_id"] += 1   # Clear the file uploader widget
    st.session_state["uploaded_image_id"] += 1  # Clear the image widget
    st.session_state["memory_thread_id"] = str(int(time())) # Create a new memory thread id, so we flush the conversation history
    st.rerun()


@logger.catch
def main():
    agent, elevenlabs_client = initialize_app()
    if config["STT_FRAMEWORK"]=="whisper":
        whisper_model = load_whisper(config["WHISPER_MODEL"])
    file_uploader_area = st.empty()

    # with st.sidebar:
    if "image_for_llm" not in st.session_state:
        uploaded_file = file_uploader_area.file_uploader(
            key=st.session_state["file_uploader_id"],
            label="Upload an image",
            label_visibility="hidden",
            # max_upload_size=MAX_UPLOAD_SIZE,
            type=config["IMAGE_TYPES"],
        )
        if uploaded_file and st.session_state["stage"] == "waiting_for_upload":
            st.session_state["image_for_llm"] = process_image(uploaded_image=uploaded_file,config=config)
            st.session_state["stage"] = "chat_welcome"
            file_uploader_area = st.empty()   # Clear the file uploader widget
            st.rerun()
        
    with st.sidebar:
        if st.button(label="Restart", type="primary"):
            restart_app()

    logger.info(f"Stage: {st.session_state["stage"]}")
    logger.info(f"Follow up questions asked: {st.session_state["questions_asked"]} of {config['MAX_QUESTIONS']}")

    if st.session_state["stage"] != ["waiting_for_upload"]:
        if "image_for_display" in st.session_state:
            _, center_col, _ = st.columns([0.1,0.8,0.1])

            with center_col:
                st.image(
                    image=st.session_state["image_for_display"],
                    width="stretch"
                )

        match st.session_state["stage"]:
            case "chat_welcome":
                # with st.spinner("LLM processing"):
                logger.info("invoke_agent: welcome message")
                agent_message = invoke_agent(
                    agent,
                    create_input(agent_instruction=config["INSTRUCTION_WELCOME"], add_image=True, system_role=config["SYSTEM_ROLE"]),
                    debug=config["DEBUG"]
                )

                tts_response = text_to_speech(elevenlabs_client, agent_message, config)
                play_speech(audio_player, tts_response)

                st.session_state["stage"] = "waiting_for_user_response"
                st.rerun()

            case "waiting_for_user_response":
                # Force a unique key at each rerun, to avoid caching errors on ST Cloud
                if "mic_id" not in st.session_state:
                    st.session_state["mic_id"] = 100001
                user_response = st.audio_input(
                    key=st.session_state["mic_id"],
                    label="audio_player",
                    label_visibility="hidden",
                    sample_rate=config["SAMPLE_RATE"]
                )
                if user_response:
                    st.session_state["user_recording"] = user_response
                    if st.session_state["questions_asked"] < config["MAX_QUESTIONS"]:
                        # We can ask more follow up questions
                        st.session_state["stage"] = "agent_respond"
                    else:
                        # We're done with follow up
                        st.session_state["stage"] = "goodbye"
                    st.session_state["mic_id"] += 1
                    st.rerun()
        
            case "agent_respond":
                if not st.session_state["user_recording"]:
                    st.write("No user recording")
                    return
                if config["DEBUG"]:
                    st.write("Debug: Playback of recorded audio")
                    st.audio(st.session_state["user_recording"])
                
                stt_start = perf_counter()
                if config["STT_FRAMEWORK"]=="whisper":
                    # With Whisper
                    # Save to a temporary file first
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(st.session_state["user_recording"].getvalue())
                        tmp_file_path = tmp_file.name
                    stt_response = whisper_model.transcribe(tmp_file_path)
                    stt_text = stt_response["text"]

                elif config["STT_FRAMEWORK"] == "elevenlabs":
                    # With Elevenlabs API
                    stt_response = elevenlabs_client.speech_to_text.convert(
                        file=st.session_state["user_recording"],
                        model_id="scribe_v2", # Model to use
                    )
                    stt_text = stt_response.text

                stt_end = perf_counter()
                logger.info(f"STT text: {stt_text}")
                logger.info(f"STT time: {stt_end-stt_start} seconds")
                # End spinner

                prompt_follow_up_chat = (
                    config["INSTRUCTION_CHAT"]
                    .replace("[[questions_asked]]", str(st.session_state["questions_asked"]))
                    .replace("[[questions_remaining]]", str(config["MAX_QUESTIONS"] - st.session_state["questions_asked"]))
                )

                agent_message = invoke_agent(
                    agent=agent,
                    input=create_input(
                        agent_instruction=prompt_follow_up_chat,
                        user_message=stt_text,
                        add_image=False
                    ),
                    debug=config["DEBUG"]
                )                

                tts_response = text_to_speech(elevenlabs_client, agent_message, config)
                play_speech(audio_player, tts_response)

                st.session_state["questions_asked"] += 1
                st.session_state["stage"] = "waiting_for_user_response"
                st.rerun()

            case "goodbye":
                logger.info("invoke_agent: goodbye")
                agent_message = invoke_agent(agent,create_input(agent_instruction=config["INSTRUCTION_GOODBYE"], add_image=False), debug=config["DEBUG"])
                tts_response = text_to_speech(elevenlabs_client, agent_message, config)
                play_speech(audio_player, tts_response)
                restart_app()
        
            
if __name__ == "__main__":
    main()
