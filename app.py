import os
import platform
import sys
import tempfile
from time import perf_counter, time

import langchain
from langchain.messages import AIMessage
from loguru import logger
import streamlit as st
import streamlit.components.v1 as components

from modules.audio import  load_whisper, text_to_speech, play_speech
from modules.llm import load_agent, create_input, invoke_agent
from modules.stages import initialize_app, display_upload_screen, restart_app
from modules.utils import load_config

# Get logs to show up on ST Cloud
if platform.system == "Linux":
    sys.stdout.flush()

# Must come before the first st. content call, so since audio_player is a global outside main(), this must come first
st.set_page_config(page_title="Your Journey", page_icon="🤳", initial_sidebar_state="collapsed")

# Create placeholder for player widget, so we can clear as needed
# TODO: Make this not a global variable
audio_player = st.empty()

@logger.catch
def main():
    config = load_config()

    agent, elevenlabs_client = initialize_app(config)
    if config["STT_FRAMEWORK"] == "whisper":
        whisper_model = load_whisper(config["WHISPER_MODEL"])

    with st.sidebar:
        if st.button(label="Restart", type="primary"):
            restart_app()

    logger.info(f"Stage: {st.session_state['stage']}")
    logger.info(f"Follow up questions asked: {st.session_state['questions_asked']} of {config['MAX_QUESTIONS']}")

    if st.session_state["stage"] == "waiting_for_upload":
        display_upload_screen(config)

    else:
        if "image_for_display" in st.session_state:
            _, center_col, _ = st.columns([0.1, 0.8, 0.1])
            with center_col:
                st.image(
                    image=st.session_state["image_for_display"],
                    use_container_width=True
                )

        match st.session_state["stage"]:
            case "chat_welcome":
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
                        st.session_state["stage"] = "agent_respond"
                    else:
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
                if config["STT_FRAMEWORK"] == "whisper":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(st.session_state["user_recording"].getvalue())
                        tmp_file_path = tmp_file.name
                    stt_response = whisper_model.transcribe(tmp_file_path)
                    stt_text = stt_response["text"]

                elif config["STT_FRAMEWORK"] == "elevenlabs":
                    stt_response = elevenlabs_client.speech_to_text.convert(
                        file=st.session_state["user_recording"],
                        model_id="scribe_v2",
                    )
                    stt_text = stt_response.text

                stt_end = perf_counter()
                logger.info(f"STT text: {stt_text}")
                logger.info(f"STT time: {stt_end - stt_start} seconds")

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
                agent_message = invoke_agent(
                    agent,
                    create_input(agent_instruction=config["INSTRUCTION_GOODBYE"], add_image=False),
                    debug=config["DEBUG"]
                )
                tts_response = text_to_speech(elevenlabs_client, agent_message, config)
                play_speech(audio_player, tts_response)
                restart_app()


if __name__ == "__main__":
    main()