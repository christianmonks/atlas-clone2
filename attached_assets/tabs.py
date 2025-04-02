import streamlit as st
from components.product_overview import render_product_overview
from components.command_center import render_command_center
from components.market_ranking import render_market_ranking
from components.matched_markets import render_matched_markets
from components.product_quick_start import render_product_quick_start
from components.power_analysis import render_power_analysis

def render_tabs():
    """
    Renders the main tab interface for the Streamlit application, consisting of
    Product Overview, Matched Market Command Center, Market Rankings & Insights,
    and Matched Markets.

    Each tab contains components for specific functionalities, and the third
    and fourth tabs are conditionally displayed based on the existence of
    `mm` in the session state (ensuring that data has been uploaded).
    """
    # Define the tabs for the interface
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "**Product Overview**",
        "**Quick Start**",
        "**Matched Market Command Center**",
        "**Market Rankings & Insights**",
        "**Matched Markets**",
        "**Power Analysis**"
    ])

    # Tab 0: Render Product Overview
    with tab0:
        render_product_overview()

    # Tab 1: Quick Start
    with tab1:
        render_product_quick_start()

    # Tab 2: Render Matched Market Command Center
    with tab2:
        render_command_center()

    # Tab 3: Render Market Rankings & Insights (only if 'mm' exists in session state)
    with tab3:
        if 'mm' in st.session_state:
            render_market_ranking()
        else:
            st.error("Please Return to the Previous Tab and Upload Audience and KPI Data", icon="🚨")

    # Tab 4: Render Matched Markets (only if 'mm' exists in session state)
    with tab4:
        if 'mm' in st.session_state:
            render_matched_markets()
        else:
            st.error("Please Return to the Previous Tab and Upload Audience and KPI Data", icon="🚨")
    with tab5:
        if 'mm' in st.session_state:
            render_power_analysis()
        else:
            st.error("Please Return to the Previous Tab and Upload Audience and KPI Data", icon="🚨")
