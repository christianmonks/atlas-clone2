import math
import hmac
import streamlit as st
from PIL import Image

def add_image(name="front_page", scale=1):
    im = Image.open(f"./image/{name}.png")
    w, h = im.size
    return im.resize((int(math.floor(w * scale)), int(math.floor(h * scale))))

def check_password(page_name):
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
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
