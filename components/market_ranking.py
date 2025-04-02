import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.constants import *

def render_market_ranking():
    """
    Renders the Market Rankings & Insights tab content.
    
    This function displays the market rankings based on the scoring algorithm,
    feature importance, and visualizations of market scores.
    """
    
    # Get required data from session state
    mm = st.session_state.get("mm")
    
    if mm is None:
        st.error("No market ranking data available. Please run the Market Ranking analysis first.")
        return
    
    # Extract ranking dataframe and feature importance
    ranking_df = mm.ranking_df
    feature_importance = mm.fi
    
    # Display Market Rankings
    st.subheader("Market Rankings")
    
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        # Show data table with market rankings
        st.dataframe(
            ranking_df[
                [mm.display_columns[0], mm.display_columns[1], TIER, SCORE, "Overall Rank", "Tier Rank"]
            ].sort_values(by=["Overall Rank"], ascending=True),
            hide_index=True,
            use_container_width=True,
        )
    
    with col2:
        # Feature importance bar chart
        st.subheader("Feature Importance")
        
        fig_importance = px.bar(
            feature_importance.sort_values(by=WEIGHT, ascending=False).head(10),
            x=WEIGHT,
            y=FEATURE,
            orientation="h",
            title="Top 10 Features by Importance",
            color=WEIGHT,
            color_continuous_scale="Viridis",
        )
        fig_importance.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Market Score Visualizations
    st.subheader("Market Score Visualizations")
    
    # Filter options for the scatter plot
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        # Dropdown for x-axis variable
        x_axis = st.selectbox(
            "**Select X-axis Variable**",
            options=[col for col in ranking_df.columns if col not in [mm.display_columns[0], mm.display_columns[1], TIER, "Overall Rank", "Tier Rank"]],
            index=ranking_df.columns.get_loc(SCORE) if SCORE in ranking_df.columns else 0,
            help="Choose the variable to display on the x-axis of the scatter plot."
        )
    
    with col2:
        # Dropdown for y-axis variable
        y_axis = st.selectbox(
            "**Select Y-axis Variable**",
            options=[col for col in ranking_df.columns if col not in [mm.display_columns[0], mm.display_columns[1], TIER, "Overall Rank", "Tier Rank"]],
            index=0,
            help="Choose the variable to display on the y-axis of the scatter plot."
        )
    
    # Create scatter plot
    fig_scatter = px.scatter(
        ranking_df,
        x=x_axis,
        y=y_axis,
        color=TIER,
        hover_name=mm.display_columns[1],
        size=SCORE,
        title=f"Market Analysis: {x_axis} vs {y_axis}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Additional Market Insights
    st.subheader("Market Insights")
    
    # Show top and bottom markets based on score
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        # Top Markets
        st.write("**Top Markets by Score**")
        top_markets = ranking_df.sort_values(by=SCORE, ascending=False).head(5)
        
        fig_top = px.bar(
            top_markets,
            x=mm.display_columns[1],
            y=SCORE,
            color=TIER,
            title="Top 5 Markets by Score",
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Bottom Markets
        st.write("**Bottom Markets by Score**")
        bottom_markets = ranking_df.sort_values(by=SCORE, ascending=True).head(5)
        
        fig_bottom = px.bar(
            bottom_markets,
            x=mm.display_columns[1],
            y=SCORE,
            color=TIER,
            title="Bottom 5 Markets by Score",
        )
        st.plotly_chart(fig_bottom, use_container_width=True)
    
    # Tier Analysis
    st.subheader("Tier Analysis")
    
    # Show average scores by tier
    tier_analysis = ranking_df.groupby(TIER)[SCORE].mean().reset_index()
    
    fig_tier = px.bar(
        tier_analysis,
        x=TIER,
        y=SCORE,
        color=TIER,
        title="Average Score by Tier",
    )
    st.plotly_chart(fig_tier, use_container_width=True)
