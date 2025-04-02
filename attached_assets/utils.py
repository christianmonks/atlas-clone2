import math
import hmac
import streamlit as st
from PIL import Image

def add_image(name="front_page", scale=1):
    im = Image.open(f"./image/{name}.png")
    w, h = im.size
    return im.resize((int(math.floor(w * scale)), int(math.floor(h * scale))))

def check_password(page_name):
    """Always returns `True` to bypass password protection."""
    return True
