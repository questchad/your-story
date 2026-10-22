import base64
import io
from loguru import logger
from PIL import Image
import streamlit as st


def process_image(uploaded_image, config):
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
    if max(img.size) > config["LLM_MAX_IMAGE_DIMENSION"]:
        # Note: Thumbnail size is maximum box, but doesn't require the image be square
        original_size = img.size
        img.thumbnail((config["LLM_MAX_IMAGE_DIMENSION"], config["LLM_MAX_IMAGE_DIMENSION"]), Image.Resampling.LANCZOS)
        logger.info(f"Resized uploaded image from {original_size} to {img.size}, enforcing max image dimension={config["LLM_MAX_IMAGE_DIMENSION"]}")

    ## Convert to JPG and, if necessary, iteratively reduce quality and dimensions to get under LLM_MAX_IMAGE_SIZE
    quality = 90
    buffer = io.BytesIO() # We'll save the image in a virtual file in memory
    
    while True:
        buffer.seek(0) #  move "cursor" back to beginning of the virtual file. Because img.save() call in the loop may have moved cursor to end of the file
        buffer.truncate() # erase the current virtual file
        img.save(buffer, format="JPEG", optimize=True, quality=quality) # Write a new one
        
        # Is the file small enough for the API?
        size_mb = buffer.tell() / (1024 * 1024)
        if size_mb <= config["LLM_MAX_IMAGE_SIZE"]:
            # We're happy!
            logger.info(f"Image is compatible with max image size: {size_mb} MB. Dimensions: {img.size}")
            break
        
        # If not, we need to optimize the file
        logger.info(f"Uploaded image is {size_mb} MB, exceeding max image size {config["LLM_MAX_IMAGE_SIZE"]} MB. Optimizing")
        quality -= 15

        if quality >= config["LLM_MIN_JPG_QUALITY"]:
            # We can try reducing the image quality
            logger.info(f"Reducing JPG quality to {quality}")
        else:
            # Image quality is too low. Increase to floor and try reducing dimensions instead
            quality = config["LLM_MIN_JPG_QUALITY"]
            current_image_dimensions = img.size
            max_current_dimension = max(current_image_dimensions)
            new_max_current_dimension = 0.75*max_current_dimension # Reduce max dimension by 75%
            img.thumbnail((new_max_current_dimension, new_max_current_dimension), Image.Resampling.LANCZOS)
            logger.info(f"Reducing image dimensions from {current_image_dimensions} to {img.size}")

    buffer.seek(0)

    image_data_for_llm = buffer.getvalue() # Read the buffer (virtual file) into a bytes object, as required by LLM APIs and st.image()

    if config["DEBUG"]:
        img.save("final_image.jpg")
        logger.info("Saved image file, with conversion if necessary, to 'final_image.jpg'")

    return image_data_for_llm

