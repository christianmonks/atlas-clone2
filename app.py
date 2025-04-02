import os
import streamlit as st

st.set_page_config(page_title="Atlas", layout="wide")
from scripts.utils import add_image, check_password
from components.tabs import render_tabs

# Set page configuration
# Manage page visibility with st_pages
from st_pages import hide_pages, show_pages, Page

# Check for password and configure page visibility
if check_password("Home"):
    show_pages([Page(path="app.py", name="Home")])  # Show home page if password is correct
else:
    hide_pages(["Home"])  # Hide home page if password is incorrect
    st.stop()  # Stop execution if password check fails

# Add logo to Streamlit app
image = add_image()  # Get the logo image using utility function
col1, col2, col3 = st.columns([1, 0.5, 1])  # Create columns for layout
with col2:
    st.image(image, width=450, use_column_width=True)  # Display logo at specified width

# Welcome statement
col1, col2, col3 = st.columns([1, 9, 1])  # Create columns for welcome message
with col2:
    # Display welcome header
    st.markdown(
        "<h3 style='text-align: center;'> 👋  Welcome to Atlas: The Matched Market Testing Suite </h3>",
        unsafe_allow_html=True,
    )

with col2:
    st.write("")  # Add an empty line for spacing
    # Display welcome text with a description of the tool
    st.write(
        """
        **Atlas is a matched market testing tool designed to craft potent market testing strategies tailored for 
        advertisers and brands. Utilizing cutting-edge AI and machine learning technologies, it discerns the interplay 
        between business KPIs and various demographic, economic, and media variables. By identifying key factors and 
        quantifying their impact on business KPIs, Atlas empowers informed decision-making.** 
        """
    )

with col2:
    render_tabs()
