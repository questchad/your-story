
import base64
import io
import os
import platform
import random
import re
import sys
import tempfile
from time import perf_counter, sleep, time

from elevenlabs.client import ElevenLabs
import langchain
from langchain.agents import create_agent
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage
from loguru import logger
from mutagen.mp3 import MP3
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import whisper

# Get logs to show up on ST Cloud
if platform.system == "Linux":
    sys.stdout.flush()


MAX_QUESTIONS = 8
TEMPERATURE = 0.5 # 1.0 Recommended for Gemma 4

# LLM_PROVIDER = "ollama"
# LLM_ID = "gemma4:e2b" # "gemma4:e2b" #"gemma3:1b-it-qat"
LLM_PROVIDER = "anthropic"
LLM_ID = "claude-haiku-4-5-20251001" # "claude-sonnet-4-6"  "claude-sonnet-4-5-20250929"  

# Gemma4: begin with <|think|> to enable thinking
SYSTEM_ROLE = """
    You are a kind, empathic college interviewer. You're interviewing a high school student to learn more about an interest or activity they've shared. 
    
    ## TONE
    You speak briefly: never reply with more than 2 sentences. 
    
    You're down to earth and kind but you always remain professional, calm. Not overly excited. Not emotive: don't laugh, chuckle, sigh etc.

    ## ROLE
    You're here to ask questions, not to advise the student, nor is your primary role to encourage them. 
        You do not give insight or hints about the student's competitiveness for college. (For example, you NEVER say things like 'colleges love it when an applicant has <a certain kind of experience or credential>".) 
    """ 
INSTRUCTION_WELCOME = """
    1. Welcome the user to this activity with "Hey thanks for doing this activity and uploading your image"
    2. The student has attached a photo or file showing 'something they could talk about for hours'. Guess from the image what that's the topic they could talk for hours about. 
    3. Then ask, "Is that right?"
"""

INSTRUCTION_CHAT = """
Based on the attachment and conversation to date, ask the user one brief follow up question to learn more about their interest/activity.

As is natural in the conversation, follow up to learn on opportunities to learn more about the interest/activity they shared in each of the following domains.

Your questions should cover the following domains, in sequence:
    1. Degree of significance/authenticity
    2. Level of engagement
    3. Leadership
    4. Accomplishments
    5. Challenges faced

If a domain is not relevant, skip it (e.g. if the student shared an interest like their favorite movie, there's probably nothing to follow up on about Leadership).

Below are sample questions you can choose from for each domain, as well as the goal for what you're trying to get at when asking about that domain. (Don't share your goals with the user.) 

Be efficient. You only need to gather enough information to answer the goal for that domain. Try to ask fewer than 2 questions about each domain, then move on to the next domain. You have asked [[questions_asked]] questions so far. You have [[questions_remaining]] questions remaining to cover the domains.

## 1. Significance/authenticity: 
GOAL: Discern if this interest/activity is a passion (e.g. core to their life or identity) or more casual/perfunctory (e.g. just doing to get the NHS hours).
    - Why did they choose to share this interest? 
    - What was their motivation for first getting involved in this activity / engaging with this interest?
    - Has participation led to personal growth?

## 2. Engagement
GOAL: Discern if this topic a casual interest/light hobby vs a rigorous passion they've dedicated part of their life to
    - How engaged are they in this interest/activity? What's their time commitment?
    - For how long have they been engaged in this interest? (Months? Years?)

## 3. Leadership
GOAL: Identify if the interest/activity demonstrates proven leadership qualities.
    - Any leadership or founding roles?

## 4. Accomplishments
GOAL: Identify if the interest/activity has led to any notable achievements or accolades.
    - Any accomplishments from their engagement in this activity? It could be traditional awards or recognition, meeting personal goals, social or business impact etc

## 5. Challenges faced
GOAL: Learn how the student responds to challenges.
    - Have they faced any challenges while engaged in this activity? 
    - How did they respond to those challenges?
"""

INSTRUCTION_GOODBYE = """
    Tell the student that you've reached the end of your questions, so that's the end of the conversation. Thank them for taking the time to share their interest (which you can mention), and wish them the best.
"""

WHISPER_MODEL = "small.en" # "tiny.en" # "turbo" # whisper.available_models()
SAMPLE_RATE = 16000 # For numpy conversion
VOICE_ID = "DODLEQrClDo8wCz460ld" # A Lauren # "WZlYpi1yf6zJhNWXih74" # A Hope "4U2riIdynwG5mhrIQRWF" # Maverick #
STT_FRAMEWORK = "elevenlabs" #"whisper"
STT_MODEL = "eleven_flash_v2_5"  # https://github.com/elevenlabs/elevenlabs-python?tab=readme-ov-file#main-models
# PIL-accepted image types
IMAGE_TYPES = [".avif",".avifs",".blp",".bmp",".bufr",".cur",".dcx",".dds",".dib",".eps",".ps",".fit",".fits",".flc",".fli",".ftc",".ftu",".gbr",".gif",".grib",".h5",".hdf",".icns",".ico",".im",".iim",".jfif",".jpe",".jpeg",".jpg",".j2c",".j2k",".jp2",".jpc",".jpf",".jpx",".msp",".pcd",".pcx",".pxr",".apng",".png",".pbm",".pfm",".pgm",".pnm",".ppm",".psd",".qoi",".bw",".rgb",".rgba",".sgi",".ras",".icb",".tga",".vda",".vst",".tif",".tiff",".webp",".emf",".wmf",".xbm",".xpm"]
LLM_MAX_IMAGE_SIZE = 3 # Anthropic's max image size for Haiku is 5MB, and it errored out on an image that's 4.7 MB on disk. TODO: Downsample
LLM_MAX_IMAGE_DIMENSION = 2048
LLM_MIN_JPG_QUALITY = 30 # Don't compress an image below this size
DEBUG = bool(os.getenv("YOUR_STORY_DEBUG_MODE", False)) # The env var sets debug mode for Cloud deployment; Change the default False/True to set debug mode for local usage


# Must come before the first st. content call, so since audio_player is a global outside main(), this must come first
st.set_page_config(page_title="Your Story", page_icon="🤳", initial_sidebar_state="collapsed")

# Create placeholder for player widget, so we can clear as needed
# TODO: Make this not a global variable
audio_player = st.empty()

@st.cache_resource(show_spinner=DEBUG)
def load_whisper():
    logger.info("Load whisper: Start")
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("Load whisper: Done")
    return model


@st.cache_resource(show_spinner=DEBUG)
def load_agent():
    logger.info("Load agent: Start")
    llm = init_chat_model(
        model=LLM_ID,
        model_provider=LLM_PROVIDER,
        temperature=TEMPERATURE    
    )

    # Automatic prompt caching
    # Will rarely be invoked due to token minimums https://platform.claude.com/docs/en/build-with-claude/prompt-caching#cache-limitations
    middleware = [AnthropicPromptCachingMiddleware()] if LLM_PROVIDER == "anthropic" else None
    agent = create_agent(
        llm,
        middleware= middleware,
        checkpointer=InMemorySaver(), # Short term memory
    )
    
    # For Ollama, dummy invocation to preload the local model
    if LLM_PROVIDER=="ollama":
        logger.info(f"Preloading {LLM_ID} on Ollama")
        response = agent.invoke(
            input=create_input(agent_instruction="hello"),
            config={"configurable": {"thread_id": str(int(time()))}}, # Use a unique thread ID, so it doesn't get added to message history nor conflict with another user using app
        )
    logger.info("Load agent: Done")
    return agent


@st.cache_resource()
def load_elevenlabs_client():
    return ElevenLabs(
         api_key=os.getenv("ELEVENLABS_API_KEY_DEV"),
    )


def process_image(uploaded_image):
    """Processes uploaded image for both LLM and display on screen (separately)
    
    Standardizes the image file uploaded by Streamlit file_uploader for the file type, size for LLM limits
    """

    # Load into PIL Image object
    # Also validates that this is a supported image type (any supported by PIL)
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    try:
        img = Image.open(uploaded_image)
    except Exception as e:
        st.error("Sorry, a problem occurred loading your image file. Could you try another image?")
        if "image_for_llm" in st.session_state:
            del st.session_state["image_for_llm"]
        if "image_to_display" in st.session_state:
            del st.session_state["image_to_display"]
        st.session_state["file_uploader_id"] += 1   # Clear the file uploader widget
        st.session_state["uploaded_image_id"] += 1  # Clear the image widget
    
    # Image for display: Store the original, unmodified image
    st.session_state["image_for_display"] = img

    # Image for LLM: modify for API requirements and optimal LLM interpretation
    ## Convert to JPG
    ## Strip any alpha/transparency data
    if img.mode != "RGB":
        logger.info("Converting image to RGB")
        img = img.convert("RGB")
    
    ## Downscale the image if width or height is too big
    # This is usually enough to get the image under LLM_MAX_IMAGE_SIZE 
    if max(img.size) > LLM_MAX_IMAGE_DIMENSION:
        # Note: Thumbnail size is maximum box, but doesn't require the image be square
        original_size = img.size
        img.thumbnail((LLM_MAX_IMAGE_DIMENSION, LLM_MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        logger.info(f"Resized uploaded image from {original_size} to {img.size}, enforcing LLM_MAX_IMAGE_DIMENSION={LLM_MAX_IMAGE_DIMENSION}")

    ## Convert to JPG and, if necessary, iteratively reduce quality and dimensions to get under LLM_MAX_IMAGE_SIZE
    quality = 90
    buffer = io.BytesIO() # We'll save the image in a virtual file in memory
    
    while True:
        buffer.seek(0) #  move "cursor" back to beginning of the virtual file. Because img.save() call in the loop may have moved cursor to end of the file
        buffer.truncate() # erase the current virtual file
        img.save(buffer, format="JPEG", optimize=True, quality=quality) # Write a new one
        
        # Is the file small enough for the API?
        size_mb = buffer.tell() / (1024 * 1024)
        if size_mb <= LLM_MAX_IMAGE_SIZE:
            # We're happy!
            logger.info(f"Image is compatible with LLM_MAX_IMAGE_SIZE: {size_mb} MB. Dimensions: {img.size}")
            break
        
        # If not, we need to optimize the file
        logger.info(f"Uploaded image is {size_mb} MB, exceeding LLM_MAX_IMAGE_SIZE {LLM_MAX_IMAGE_SIZE} MB. Optimizing")
        quality -= 15

        if quality >= LLM_MIN_JPG_QUALITY:
            # We can try reducing the image quality
            logger.info(f"Reducing JPG quality to {quality}")
        else:
            # Image quality is too low. Increase to floor and try reducing dimensions instead
            quality = LLM_MIN_JPG_QUALITY
            current_image_dimensions = img.size
            max_current_dimension = max(current_image_dimensions)
            new_max_current_dimension = 0.75*max_current_dimension # Reduce max dimension by 75%
            img.thumbnail((new_max_current_dimension, new_max_current_dimension), Image.Resampling.LANCZOS)
            logger.info(f"Reducing image dimensions from {current_image_dimensions} to {img.size}")

    buffer.seek(0)

    image_data_for_llm = buffer.getvalue() # Read the buffer (virtual file) into a bytes object, as required by LLM APIs and st.image()

    if DEBUG:
        img.save("final_image.jpg")
        logger.info("Saved image file, with conversion if necessary, to 'final_image.jpg'")

    return image_data_for_llm


def create_input(agent_instruction, user_message=None, add_image=False):
    messages = []

    if st.session_state.get("stage", None) == "chat_welcome":
        # Add system role to the first message
        messages.append({"role": "system", "content": SYSTEM_ROLE})

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
    

def invoke_agent(agent, input):
    if platform.system == "Linux":
        sys.stdout.flush()
    llm_start = perf_counter()
    if DEBUG:
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


def play_speech(elevenlabs_tts_response):
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


def text_to_speech(elevenlabs_client, text):
    tts_start = perf_counter()
    logger.info("TTS: start")
    tts_response = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=STT_MODEL,
    )
    tts_end = perf_counter()
    logger.info(f"TTS: {tts_end-tts_start} seconds")

    return tts_response


def initialize_app():
    logger.info("Initializing app")
    if "stage" not in st.session_state:
        st.session_state["stage"] = "waiting_for_upload"
        st.header("Please upload an image to start")

    agent = load_agent()
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
    if STT_FRAMEWORK=="whisper":
        whisper_model = load_whisper()
    file_uploader_area = st.empty()

    # with st.sidebar:
    if "image_for_llm" not in st.session_state:
        uploaded_file = file_uploader_area.file_uploader(
            key=st.session_state["file_uploader_id"],
            label="Upload an image",
            label_visibility="hidden",
            # max_upload_size=MAX_UPLOAD_SIZE,
            type=IMAGE_TYPES,
        )
        if uploaded_file and st.session_state["stage"] == "waiting_for_upload":
            st.session_state["image_for_llm"] = process_image(uploaded_file)
            st.session_state["stage"] = "chat_welcome"
            file_uploader_area = st.empty()   # Clear the file uploader widget
            st.rerun()
        
    with st.sidebar:
        if st.button(label="Restart", type="primary"):
            restart_app()

    logger.info(f"Stage: {st.session_state["stage"]}")
    logger.info(f"Follow up questions asked: {st.session_state["questions_asked"]} of {MAX_QUESTIONS}")

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
                agent_message = invoke_agent(agent,create_input(agent_instruction=INSTRUCTION_WELCOME, add_image=True))

                tts_response = text_to_speech(elevenlabs_client, agent_message)
                play_speech(tts_response)

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
                    sample_rate=SAMPLE_RATE
                )
                if user_response:
                    st.session_state["user_recording"] = user_response
                    if st.session_state["questions_asked"] < MAX_QUESTIONS:
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
                if DEBUG:
                    st.write("Debug: Playback of recorded audio")
                    st.audio(st.session_state["user_recording"])
                
                stt_start = perf_counter()
                if STT_FRAMEWORK=="whisper":
                    # With Whisper
                    # Save to a temporary file first
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(st.session_state["user_recording"].getvalue())
                        tmp_file_path = tmp_file.name
                    stt_response = whisper_model.transcribe(tmp_file_path)
                    stt_text = stt_response["text"]

                elif STT_FRAMEWORK == "elevenlabs":
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
                    INSTRUCTION_CHAT
                    .replace("[[questions_asked]]", str(st.session_state["questions_asked"]))
                    .replace("[[questions_remaining]]", str(MAX_QUESTIONS - st.session_state["questions_asked"]))
                )

                agent_message = invoke_agent(
                    agent=agent,
                    input=create_input(
                        agent_instruction=prompt_follow_up_chat,
                        user_message=stt_text,
                        add_image=False
                    )
                )                

                tts_response = text_to_speech(elevenlabs_client, agent_message)
                play_speech(tts_response)

                st.session_state["questions_asked"] += 1
                st.session_state["stage"] = "waiting_for_user_response"
                st.rerun()

            case "goodbye":
                logger.info("invoke_agent: goodbye")
                agent_message = invoke_agent(agent,create_input(agent_instruction=INSTRUCTION_GOODBYE, add_image=False))
                tts_response = text_to_speech(elevenlabs_client, agent_message)
                play_speech(tts_response)
                restart_app()
        
            
if __name__ == "__main__":
    main()
