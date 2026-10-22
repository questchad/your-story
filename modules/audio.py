import base64
import io
import os

from elevenlabs.client import ElevenLabs
from loguru import logger
from mutagen.mp3 import MP3
import streamlit as st
from time import perf_counter, sleep, time
import whisper


@st.cache_resource(show_spinner=False)
def load_whisper(whisper_model):
    logger.info("Load whisper: Start")
    model = whisper.load_model(whisper_model)
    logger.info("Load whisper: Done")
    return model


@st.cache_resource()
def load_elevenlabs_client():
    return ElevenLabs(
         api_key=os.getenv("ELEVENLABS_API_KEY_DEV"),
    )


def play_speech(audio_player, elevenlabs_tts_response):
    """Function to play a speech audio from ElevenLabs's Text-to-Speech endpoint.

    We need a solution that plays the audio both locally (when laptop = Streamlit server) and when accessed on Streamlit Community Cloud (when browser = client, and server is somewhere in the cloud).
    """
    # Convert audio to bytes
    tts_response_bytes = b"".join(elevenlabs_tts_response)

    # Calculate how long the audio is
    tts_response_duration_sec = (
        MP3(
            io.BytesIO(tts_response_bytes)
        )
        .info.length
    )
    logger.info(f"TTS: File duration {tts_response_duration_sec} sec")

    # Inject an HTML audio player, hidden controls, autoplay (start playing now)
    # To avoid audio artifacts, we add a unique identifier for this specific playback instance (so the element doesn't try to recycle the element) 
    player_id = str(int(time()))
    # CSS to make it completely invisible and take up 0px of space
    hidden_style = 'style="visibility:hidden; display:none; height:0; width:0;"'
    tts_response_base64 = base64.b64encode(tts_response_bytes).decode('utf-8')
    audio_html = f"""
        <audio id="audio_{player_id}" autoplay="true" {hidden_style}>
            <source src="data:audio/mp3;base64,{tts_response_base64}" type="audio/mp3">
        </audio>
        """
    audio_player.markdown(audio_html, unsafe_allow_html=True)

    # Since we want to play synchoronously, because a play_speech() call is followed by an st.rerun,
    # we need lock the Streamlit app for the number of seconds that the audio is playing
    # Otherwise, st.rerun() will cut off the audio
    sleep(tts_response_duration_sec + 1) # Pause the Streamlit flow while audio is playing

    # Clear the widget placeholder. Otherwise we have problems at the next run of st.audio_input, may be due to conflicts between play_speech() and next audio_input operation
    audio_player.empty()

    # Add some time for browser hardware to catch up before an st.audio_input call, otherwise it has an error
    sleep(0.5)


def text_to_speech(elevenlabs_client, text, config):
    tts_start = perf_counter()
    logger.info("TTS: start")
    tts_response = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=config["VOICE_ID"],
        model_id=config["STT_MODEL"]
    )
    tts_end = perf_counter()
    logger.info(f"TTS: {tts_end-tts_start} seconds")

    return tts_response

