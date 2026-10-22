import base64
import os
import yaml

import streamlit as st

@st.cache_data()
def load_config():
    with open("config.yml") as f:
        config = yaml.safe_load(f)
    # The env var overrides debug mode for Cloud deployment
    config["DEBUG"] = bool(
        os.getenv("YOUR_STORY_DEBUG_MODE", config.get("DEBUG", False))
    )
     
    return config


def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #05182d
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
