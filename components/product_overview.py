import streamlit as st

def render_product_overview():
    """
    Renders the 'Product Overview' section for the Matched Market Testing Suite.
    This section includes details on the features and enhanced capabilities of the suite,
    with content shown inside expanders for better readability.
    """

    # Expander for showcasing the key features of the suite
    with st.expander("**Features of the Matched Market Testing Suite**", expanded=True):
        st.markdown("""
        * **Market Prioritization Tool**: Provides a market ranking to prioritize markets for media testing and expansion.
        * **Matched Markets Tool**: Provides matched market pairs for media testing and uplift modeling.
        """)

    # Expander for showcasing the enhanced capabilities of the suite
    with st.expander("**Enhanced Capabilities of the Matched Market Testing Suite**", expanded=True):
        st.markdown("""
        * **Diverse Market Analysis**: Comprehensive assessments at Country and US DMA levels.
        * **Adaptable Metrics**: Accommodates KPIs like sales volume and Brand Equity levels.
        * **Broad Data Integration**: Seamlessly integrates 1st and 3rd-party data for enriched insights.
        """)
