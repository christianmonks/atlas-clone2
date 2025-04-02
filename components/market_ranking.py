import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.constants import *

def render_market_ranking():
    """
    Renders the Market Ranking and Insights section of the application.
    
    This component displays market rankings based on the analysis performed
    in the command center, showing sorted market scores, feature importance,
    and visualizations to help understand market performances.
    """
    # Get necessary session state variables
    mm, kpi_column = st.session_state['mm'], st.session_state['kpi_column']
    
    # Title and description
    st.markdown(
        "<h4 style='text-align: center; color: black;'>Market Ranking Based on ML Market Scores</h4>",
        unsafe_allow_html=True
    )
    
    # Market ranking table
    ranking_df = mm.ranking_df
    display_cols = [
        col for col in [
            'DMA Code', 'DMA Name', 
            'State Code', 'State Name', 
            'Country Code', 'Country Name',
            'KPI Tier', 'Score', 'Overall Rank', 'Tier Rank'
        ] if col in ranking_df.columns
    ]
    
    st.dataframe(
        ranking_df[display_cols].sort_values(by='Overall Rank', ascending=True),
        use_container_width=True,
        hide_index=True
    )
    
    # Create columns for feature importance and market score visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display feature importance
        st.markdown(
            "<h5 style='text-align: center; color: black;'>Feature Importance</h5>",
            unsafe_allow_html=True
        )
        
        st.dataframe(
            mm.fi.sort_values(by=WEIGHT, ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        # Create market score visualization
        st.markdown(
            f"<h5 style='text-align: center; color: black;'>Market Scores by {kpi_column}</h5>",
            unsafe_allow_html=True
        )
        
        # Plot market scores using Plotly
        market_name_col = [col for col in ['DMA Name', 'State Name', 'Country Name'] if col in ranking_df.columns][0]
        
        fig = px.scatter(
            ranking_df.sort_values(by='Score', ascending=False),
            x='Score',
            y=market_name_col,
            color='KPI Tier',
            size='Score',
            hover_data=['Overall Rank', 'Tier Rank'],
            title=f"Market Scores by {kpi_column}",
            height=600
        )
        
        fig.update_layout(
            xaxis_title="Market Score",
            yaxis_title="Market",
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed market insights
    with st.expander("**Detailed Market Insights**", expanded=True):
        st.markdown("""
        ### Understanding the Market Ranking
        
        The markets are ranked based on a machine learning model that evaluates the following factors:
        
        1. **Target Audience Size**: Markets with larger concentrations of your target audience receive higher scores
        2. **Demographic Variables**: Socioeconomic factors that correlate with your KPI
        3. **Past Performance**: Historical KPI data showing market potential
        
        ### How to Use This Information
        
        - **High-Scoring Markets**: Consider these for expansion or increased marketing effort
        - **Tier Analysis**: Look at markets within each performance tier for comparative insights
        - **Feature Importance**: Understand which factors most strongly influence market performance
        
        The scores provide a data-driven approach to prioritizing markets for testing and investment.
        """)
