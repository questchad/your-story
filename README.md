# Reminders
* Components used 
    * Speech to Text: ElevenLabs API (Whisper local may work too; this option was part of code copied from Passion Chat prototype) 
    * LLM: Anthropic API or Ollama (local)
    * Text to speech: ElevenLabs API
* Debug mode
    * Will add more on-screen indicators of status
    * Will save the converted image as JPG, optimized if necessary for `MAX_IMAGE_SIZE` and `MAX_IMAGE_DIMENSION`, to `final_image.jpg`
* Global fonts etc: in `.streamlit/config.toml`

# Run locally
1. Install `uv` on your machine if necessary
2. In parent directory, run `uv sync`.
3. In this directory, in a terminal, run `uv run streamlit run app.py`. 
4. Open the link that's printed in the terminal in your browser.

Notes  
* Logs show up in the terminal
* To run in DEBUG mode locally, change the False to True in this line in app.py. `DEBUG = bool(os.getenv("YOUR_STORY_DEBUG_MODE", False))`


# Deploy on Streamlit Community Cloud

* To use on the cloud, ensure `STT_FRAMEWORK = "elevenlabs"`. Can only use Whisper locally.
* The app is set up to deploy to Streamlit Community Cloud. The free plan allows 1 private app.
* The deployment is synchronized to the GitHub repo. When a new PR is merged to `main`, the deployed web app is updated. So if you want to change the deployed version, just merge a PR!.
* To manage the deployed web app, visit https://share.streamlit.io/. 
    - Always ensure the app is Private, not Public. Only specified people should be able to access
    - You determine who has access to the app
    - The app admin includes the secrets needed `ANTHROPIC_API_KEY` and `ELEVENLABS_API_KEY_DEV` under the UI's app Settings/Secrets, which exposes environment variables just like the local app uses
    - To run in DEBUG mode, change the secret in the Admin UI `YOUR_STORY_DEBUG_MODE`
    - Logs show up in the right sidebar, click Manage app on bottom right