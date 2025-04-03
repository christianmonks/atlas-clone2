import math
import hmac
import streamlit as st
from PIL import Image

def add_image(name="front_page", scale=1):
    """
    Load and resize an image from the image directory.
    
    Args:
        name (str): Base name of the image file (default: "front_page")
        scale (float): Scale factor for resizing the image (default: 1)
        
    Returns:
        PIL.Image: The resized image
    """
    try:
        im = Image.open(f"./image/{name}.png")
        w, h = im.size
        return im.resize((int(math.floor(w * scale)), int(math.floor(h * scale))))
    except FileNotFoundError:
        # Create a blank image if file not found
        im = Image.new('RGB', (450, 200), color = (255, 255, 255))
        return im

def check_password(page_name):
    """
    Returns `True` if the user had the correct password.
    
    Args:
        page_name (str): The name of the current page
        
    Returns:
        bool: True if password is correct, False otherwise
    """

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Use hardcoded 'password' instead of streamlit secrets
        if hmac.compare_digest(st.session_state["password"], "password"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    if page_name == "Home":
        # Show input for password only on Home page.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
    else:
        st.error("Enter your password on the home page to access")

    if "password_correct" in st.session_state:
        st.error("Password incorrect")
    return False
