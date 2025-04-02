import streamlit as st
import pandas as pd
import os
import json
import re
from scripts.constants import *
from scripts.matched_market import MatchedMarketScoring

def render_command_center():
    """
    Renders the Matched Market Command Center which serves as the main control area
    for configuring and running market analysis.
    
    This function allows users to:
    1. Select the country and market level
    2. Choose target audiences
    3. Upload client KPI and optional client data
    4. Configure KPI selection and date granularity
    5. Select additional data sources
    6. Run the market ranking analysis
    """
    
    # Initialize session state variables if they don't exist
    if "mm" not in st.session_state:
        st.session_state["mm"] = None
    if "mm1" not in st.session_state:
        st.session_state["mm1"] = None
    if "feature_importance" not in st.session_state:
        st.session_state["feature_importance"] = None
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "kpi_df" not in st.session_state:
        st.session_state["kpi_df"] = None
        
    # Create columns for layout
    col1, col2 = st.columns([1, 2], gap="small")
    
    with col1:
        # Country and Market Level Selection
        st.subheader("Country and Market Level Selection")
        country_level = st.selectbox(
            "**Select Country Level**",
            options=MARKET_LEVELS,
            help="Choose the geographical level for your market analysis."
        )
        
        # Set market code and name based on the selected country level
        if country_level == "US DMA":
            market_code = DMA_CODE
            market_name = DMA_NAME
        elif country_level == "US State":
            market_code = STATE_CODE
            market_name = STATE_NAME
        elif country_level == "BR Municipality":
            market_code = "Municipality Code"
            market_name = "Municipality Name"
        elif country_level == "MX Municipality":
            market_code = "Municipality Code"
            market_name = "Municipality Name"
        else:
            market_code = COUNTRY_CODE
            market_name = COUNTRY_NAME
        
        # Store market code and name in session state
        st.session_state["market_code"] = market_code
        st.session_state["market_name"] = market_name
        st.session_state["market_level"] = country_level
        
        # Target Audience Selection
        st.subheader("Target Audience Selection")
        st.write("Choose up to three target audiences for your analysis.")
        
        # Dummy audience options based on country level
        audience_options = ["All", "F18+", "M18+", "P18-34", "P35-54", "P55+"]
        
        audience_1 = st.selectbox(
            "**Primary Audience**",
            options=audience_options,
            index=0,
            help="Select your primary target audience."
        )
        
        audience_2 = st.selectbox(
            "**Secondary Audience** (Optional)",
            options=["None"] + audience_options,
            index=0,
            help="Select an optional secondary target audience."
        )
        
        audience_3 = st.selectbox(
            "**Tertiary Audience** (Optional)",
            options=["None"] + audience_options,
            index=0,
            help="Select an optional tertiary target audience."
        )
        
        # Collect selected audiences
        audience_column = [audience_1]
        if audience_2 != "None":
            audience_column.append(audience_2)
        if audience_3 != "None":
            audience_column.append(audience_3)
        
        st.session_state["audience_column"] = audience_column
        
    with col2:
        # Data Upload Section
        st.subheader("Data Uploader")
        
        # Client KPI Data Upload
        st.write("Upload your Client KPI Data (required)")
        uploaded_kpi = st.file_uploader(
            "Upload Client KPI Data (CSV)",
            type=["csv"],
            help="Upload a CSV file containing your KPI data with appropriate format."
        )
        
        # Client Optional Data Upload
        st.write("Upload Optional Client-Specific Data")
        uploaded_client_data = st.file_uploader(
            "Upload Client-Specific Data (CSV)",
            type=["csv"],
            help="Upload a CSV file containing additional client-specific data."
        )
        
        # Sample Data Section
        with st.expander("**Sample Data Preview**"):
            st.write("This will show a preview of your uploaded data.")
            
            # Show KPI Data preview if uploaded
            if uploaded_kpi is not None:
                kpi_df = pd.read_csv(uploaded_kpi)
                st.write("**KPI Data Preview**")
                st.dataframe(kpi_df.head())
                
                # Store KPI DataFrame in session state
                st.session_state["kpi_df"] = kpi_df
                
                # Extract KPI columns for later selection
                kpi_columns = [col for col in kpi_df.columns if "KPI_" in col]
                st.session_state["kpi_columns"] = kpi_columns
                
                # Identify date column if it exists
                date_column = "Date" if "Date" in kpi_df.columns else None
                st.session_state["date_column"] = date_column
            
            # Show Client Data preview if uploaded
            if uploaded_client_data is not None:
                client_df = pd.read_csv(uploaded_client_data)
                st.write("**Client Data Preview**")
                st.dataframe(client_df.head())
                
                # Extract client columns for analysis
                client_columns = [col for col in client_df.columns if "CLIENT_" in col]
                st.session_state["client_columns"] = client_columns
                
                # If both KPI and client data are uploaded, merge them
                if uploaded_kpi is not None:
                    # Merge on market column
                    merged_df = pd.merge(kpi_df, client_df, on="Market", how="left")
                    st.session_state["df"] = merged_df
                else:
                    st.session_state["client_columns"] = []
            else:
                # If no client data, use only KPI data
                if uploaded_kpi is not None:
                    st.session_state["df"] = kpi_df
                    st.session_state["client_columns"] = []
    
    # KPI Selection and Configuration Section
    st.subheader("KPI Selection and Configuration")
    
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    
    with col1:
        # KPI Selection dropdown
        if "kpi_columns" in st.session_state and st.session_state["kpi_columns"]:
            kpi_column = st.selectbox(
                "**Select KPI**",
                options=st.session_state["kpi_columns"],
                help="Choose the KPI to use for market analysis."
            )
            st.session_state["kpi_column"] = kpi_column
        else:
            st.warning("Upload KPI data to select a KPI.")
    
    with col2:
        # Date Granularity selection
        date_granularity = st.selectbox(
            "**Date Granularity**",
            options=["Daily", "Weekly", "Monthly", "No Date"],
            help="Select the time granularity for your KPI data."
        )
        st.session_state["date_granularity"] = date_granularity
    
    with col3:
        # Data sources selection
        data_sources = st.multiselect(
            "**Data Sources**",
            options=["Census Demographics", "Media Consumption", "Economic Indicators"],
            default=["Census Demographics"],
            help="Select additional data sources to enrich your analysis."
        )
        st.session_state["data_sources"] = data_sources
    
    # Covariate Selection Section based on country level
    st.subheader("Covariate Selection")
    
    # Get default columns based on country level
    country_key = country_level.replace(" ", "_")
    default_cols = DEFAULT_COLUMNS.get(country_key, [])
    
    # Allow user to select covariates
    cov_columns = st.multiselect(
        "**Select Covariates**",
        options=default_cols,
        default=default_cols[:4] if default_cols else [],
        help="Select covariates to include in your market analysis."
    )
    st.session_state["cov_columns"] = cov_columns
    
    # Add columns that indicate spend for removal from scoring
    spend_cols = st.multiselect(
        "**Select Spend Columns to Exclude from Scoring**",
        options=st.session_state.get("client_columns", []),
        help="Select columns that represent spend which should be excluded from market scoring."
    )
    st.session_state["spend_cols"] = spend_cols
    
    # Run Market Ranking button
    if st.button("**Confirm and Run Market Ranking 🏃‍➡**"):
        with st.spinner("Running Market Ranking Analysis..."):
            if "df" in st.session_state and "kpi_df" in st.session_state and st.session_state["df"] is not None and st.session_state["kpi_df"] is not None:
                # Add a tier column based on KPI values if it doesn't exist
                if TIER not in st.session_state["df"].columns:
                    # Create a simple tier classification based on KPI percentiles
                    df = st.session_state["df"]
                    kpi_col = st.session_state["kpi_column"]
                    
                    # Create tiers based on KPI percentiles
                    df[TIER] = pd.qcut(df[kpi_col], q=3, labels=["Low", "Medium", "High"])
                    st.session_state["df"] = df
                
                # Run the Matched Market Scoring algorithm
                try:
                    mm = MatchedMarketScoring(
                        df=st.session_state["df"],
                        kpi_df=st.session_state["kpi_df"],
                        audience_columns=st.session_state["audience_column"],
                        client_columns=st.session_state.get("client_columns", []),
                        covariate_columns=st.session_state["cov_columns"],
                        display_columns=[st.session_state["market_code"], st.session_state["market_name"]],
                        market_column=st.session_state["market_code"],
                        date_granularity=st.session_state["date_granularity"],
                        kpi_column=st.session_state["kpi_column"],
                        scoring_removed_columns=st.session_state["spend_cols"],
                        run_model=True
                    )
                    
                    # Store results in session state
                    st.session_state["mm"] = mm
                    st.session_state["mm1"] = mm
                    st.session_state["feature_importance"] = mm.feature_importance
                    
                    # Show success message
                    st.success("Market Ranking completed successfully! Navigate to the 'Market Rankings & Insights' tab to view results.")
                    
                except Exception as e:
                    st.error(f"Error running Market Ranking: {str(e)}")
            else:
                st.error("Please upload KPI data first.")
