import os
from time import time
import yaml

from loguru import logger
import streamlit as st

from modules.audio import load_elevenlabs_client
from modules.image import process_image
from modules.llm import load_agent
from modules.utils import set_background

def initialize_app(config):
    logger.info("Initializing app")
    if "stage" not in st.session_state:
        st.session_state["stage"] = "waiting_for_upload"

    agent = load_agent(config)
    elevenlabs_client = load_elevenlabs_client()

    if "questions_asked" not in st.session_state:
        st.session_state["questions_asked"] = 0

    # Initialize 4 identifiers for uploaded images and file uploader widgets
    for i in range(1, 5):
        if f"uploaded_image_id_{i}" not in st.session_state:
            st.session_state[f"uploaded_image_id_{i}"] = 1000 + i
        if f"file_uploader_id_{i}" not in st.session_state:
            st.session_state[f"file_uploader_id_{i}"] = 10000 + i
    if "memory_thread_id" not in st.session_state:
        st.session_state["memory_thread_id"] = str(int(time())) # Create a new ID unique to this user's interaction
    

    set_background()

    return agent, elevenlabs_client


def restart_app():
    logger.info("Restarting app")
    st.cache_data.clear()
    
    # Keys to reset
    resettable_keys = ["stage", "questions_asked", "user_recording"]
    for i in range(1, 5):
        resettable_keys.extend([
            f"image_for_llm_{i}", 
            f"image_for_display_{i}",
            f"processed_file_id_{i}"
        ])

    for key in resettable_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Increment widget IDs to clear uploaders
    for i in range(1, 5):
        st.session_state[f"uploaded_image_id_{i}"] += 10 # Clear the image widget
        st.session_state[f"file_uploader_id_{i}"] += 10  # Clear the file uploader widget the image widget
        
    st.session_state["memory_thread_id"] = str(int(time()))
    st.rerun()


def display_upload_screen(config):
    st.markdown(
        # Intro page header
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');
        </style>

        <div style="text-align: center">
        <div style="font-family: 'Great Vibes', cursive; font-size: 124px; font-weight: normal; margin-top: 10px; margin-bottom: 25px;">
            Your Journey
        </div>
        <div style="font-size: 22px">Please upload photos from different times in your life.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.space(size="large")

    # 2x2 Grid setup
    row1_col1, row1_col2 = st.columns(2, gap="large", border=False)
    row2_col1, row2_col2 = st.columns(2, gap="large", border=False)
    grid_cols = [row1_col1, row1_col2, row2_col1, row2_col2]

    for idx, (col, caption) in enumerate(zip(grid_cols, config["IMAGE_CAPTIONS"]), start=1):
        with col:
            # Caption over each uploader
            st.markdown(body=f"""<div style="font-size: 28px; margin-bottom:22px">{caption}</div>""", unsafe_allow_html=True, text_alignment="center")

            # Create a place to display the uploaded image
            image_outer_container = col.container(
                horizontal_alignment="center",
                border=False,
            )
            image_container =  image_outer_container.container(
                height=150,
                horizontal_alignment="center",
                vertical_alignment="center",
                border=False 
            )


            # File uploader
            st.markdown(
                """
                <!-- Hide hints for '200MB' and file types -->

                <style>
                div[data-testid="stFileUploaderDropzoneInstructions"],
                div[data-testid="stFileUploaderDropzone"] {
                    display: none !important;
                }
                </style>

                <!-- Center the upload buttons -->
                <style>
                            [data-testid="stFileUploaderDropzone"] {
                    display: flex !important;
                    justify-content: center !important;
                    align-items: center !important;
                }
                </style>

                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                label=caption,
                key=st.session_state[f"file_uploader_id_{idx}"],
                type=config["IMAGE_TYPES"],
                label_visibility="hidden"
            )
            
            st.space("large")

            if uploaded_file:
                file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
                if st.session_state.get(f"processed_file_id_{idx}") != file_id:
                    st.session_state[f"image_for_llm_{idx}"] = process_image(
                        uploaded_image=uploaded_file, idx=idx, config=config
                    )
                    st.session_state[f"processed_file_id_{idx}"] = file_id
            else:
                if f"image_for_llm_{idx}" in st.session_state:
                    del st.session_state[f"image_for_llm_{idx}"]
                if f"processed_file_id_{idx}" in st.session_state:
                    del st.session_state[f"processed_file_id_{idx}"]
    
            # Update the displayed image above
            image_container.markdown(
                """
                <!-- Make the image fit without scrollbars -->
                <style>
                div[data-testid="stImageContainer"] img {
                    border: 2px;
                    max-height: 130px;
                    width: auto !important;
                    max-width: 100%;
                    border-radius: 18px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            if f"image_for_display_{idx}" in st.session_state:
                image_container.image(
                    image=st.session_state[f"image_for_display_{idx}"], 
                )

    st.space(size="medium")

    
    # Continue button
    _, _, center, _, _ = st.columns([1,1,1,1,1])
    if center.button("Continue", width=150):
        has_at_least_one_image = any(f"image_for_llm_{i}" in st.session_state for i in range(1, 5))
        if has_at_least_one_image:
            st.session_state["stage"] = "chat_welcome"
            st.rerun()
        else:
            st.warning("Please upload at least one photo before continuing.")