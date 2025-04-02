import streamlit as st
import pandas as pd
import numpy as np
import os
from scripts.matched_market import MatchedMarketScoring
from scripts.constants import *

def render_command_center():
    """
    Renders the Matched Market Command Center interface, allowing users to configure
    market analysis parameters, upload data, and run market ranking analysis.
    """
    # Initialize session state variables if they don't exist
    if 'market_level' not in st.session_state:
        st.session_state['market_level'] = None
    if 'audience_column' not in st.session_state:
        st.session_state['audience_column'] = []
    if 'client_columns' not in st.session_state:
        st.session_state['client_columns'] = []
    if 'cov_columns' not in st.session_state:
        st.session_state['cov_columns'] = []
    if 'market_code' not in st.session_state:
        st.session_state['market_code'] = None
    if 'market_name' not in st.session_state:
        st.session_state['market_name'] = None
    if 'date_column' not in st.session_state:
        st.session_state['date_column'] = None
    if 'date_granularity' not in st.session_state:
        st.session_state['date_granularity'] = 'Daily'
    if 'kpi_column' not in st.session_state:
        st.session_state['kpi_column'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'kpi_df' not in st.session_state:
        st.session_state['kpi_df'] = None
    if 'spend_cols' not in st.session_state:
        st.session_state['spend_cols'] = []
    if 'mm' not in st.session_state:
        st.session_state['mm'] = None
    if 'mm1' not in st.session_state:
        st.session_state['mm1'] = None
    if 'feature_importance' not in st.session_state:
        st.session_state['feature_importance'] = None
    
    # Set up tabs for configuration
    with st.expander("**Market & Audience Selection**", expanded=True):
        # Create two columns for market level selection and audience selection
        col1, col2 = st.columns(2)
        
        with col1:
            # Select market level
            market_level = st.selectbox(
                "**Select Market Level**",
                options=MARKET_LEVELS,
                help="Choose the geographic level at which you want to analyze markets."
            )
            st.session_state['market_level'] = market_level
            
            # Set up market code and name based on selection
            if market_level == "US DMA":
                market_code = "DMA Code"
                market_name = "DMA Name"
            elif market_level == "US State":
                market_code = "State Code"
                market_name = "State Name"
            elif market_level == "BR Municipality" or market_level == "MX Municipality":
                market_code = "Market"
                market_name = "Market Name"
            else:
                market_code = "Country Code"
                market_name = "Country Name"
                
            st.session_state['market_code'] = market_code
            st.session_state['market_name'] = market_name
        
        with col2:
            # Target audience selection (placeholder)
            target_audience = st.multiselect(
                "**Select Target Audience**",
                options=["Adults 18+", "Men 18+", "Women 18+", "Adults 18-34", "Women 18-34", "Men 18-34", "Adults 35+", "Women 35+", "Men 35+"],
                max_selections=3,
                help="Choose up to three target audiences for your analysis."
            )
            
            # Convert audience selections to column names
            audience_columns = []
            for audience in target_audience:
                if audience == "Adults 18+":
                    audience_columns.append("P18+")
                elif audience == "Men 18+":
                    audience_columns.append("M18+")
                elif audience == "Women 18+":
                    audience_columns.append("F18+")
                elif audience == "Adults 18-34":
                    audience_columns.append("P18-34")
                elif audience == "Women 18-34":
                    audience_columns.append("F18-34")
                elif audience == "Men 18-34":
                    audience_columns.append("M18-34")
                elif audience == "Adults 35+":
                    audience_columns.append("P35+")
                elif audience == "Women 35+":
                    audience_columns.append("F35+")
                elif audience == "Men 35+":
                    audience_columns.append("M35+")
            
            st.session_state['audience_column'] = audience_columns
    
    with st.expander("**Data Uploader**", expanded=True):
        # Create columns for data upload
        col1, col2 = st.columns(2)
        
        with col1:
            # Client KPI data upload
            kpi_data = st.file_uploader(
                "**Upload Client KPI Data (required)**",
                type=["csv", "xlsx"],
                help="Upload a CSV or Excel file containing your KPI data with the required schema."
            )
            
            if kpi_data is not None:
                try:
                    # Read the data based on file type
                    if kpi_data.name.endswith('.csv'):
                        kpi_df = pd.read_csv(kpi_data)
                    else:
                        kpi_df = pd.read_excel(kpi_data)
                    
                    # Show success message and data preview
                    st.success(f"Successfully loaded KPI data with {kpi_df.shape[0]} rows and {kpi_df.shape[1]} columns")
                    st.dataframe(kpi_df.head(5), use_container_width=True)
                    
                    # Store in session state
                    st.session_state['kpi_df'] = kpi_df
                except Exception as e:
                    st.error(f"Error loading KPI data: {str(e)}")
        
        with col2:
            # Client specific data upload (optional)
            client_data = st.file_uploader(
                "**Upload Client Specific Data (optional)**",
                type=["csv", "xlsx"],
                help="Optionally upload a CSV or Excel file with additional client-specific data."
            )
            
            if client_data is not None:
                try:
                    # Read the data based on file type
                    if client_data.name.endswith('.csv'):
                        client_df = pd.read_csv(client_data)
                    else:
                        client_df = pd.read_excel(client_data)
                    
                    # Show success message and data preview
                    st.success(f"Successfully loaded client data with {client_df.shape[0]} rows and {client_df.shape[1]} columns")
                    st.dataframe(client_df.head(5), use_container_width=True)
                    
                    # Extract client columns
                    client_cols = [col for col in client_df.columns if col.startswith('CLIENT_')]
                    if client_cols:
                        st.session_state['client_columns'] = client_cols
                    else:
                        st.warning("No CLIENT_ columns found in the uploaded data")
                    
                    # Store client data for later use
                    st.session_state['client_df'] = client_df
                except Exception as e:
                    st.error(f"Error loading client data: {str(e)}")
    
    with st.expander("**KPI & Date Configuration**", expanded=True):
        # KPI and date configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # KPI selection
            if 'kpi_df' in st.session_state and st.session_state['kpi_df'] is not None:
                kpi_columns = [col for col in st.session_state['kpi_df'].columns if col.startswith('KPI_')]
                if kpi_columns:
                    kpi_column = st.selectbox(
                        "**Select KPI for Analysis**",
                        options=kpi_columns,
                        help="Choose the Key Performance Indicator to analyze."
                    )
                    st.session_state['kpi_column'] = kpi_column
                else:
                    st.warning("No KPI_ columns found in the uploaded data")
            else:
                st.info("Please upload KPI data to select KPI columns")
        
        with col2:
            # Date granularity selection
            if 'kpi_df' in st.session_state and st.session_state['kpi_df'] is not None:
                # Check if Date column exists
                if 'Date' in st.session_state['kpi_df'].columns:
                    st.session_state['date_column'] = 'Date'
                    date_granularity = st.selectbox(
                        "**Date Granularity**",
                        options=["Daily", "Weekly"],
                        help="Specify whether your data is daily or weekly."
                    )
                    st.session_state['date_granularity'] = date_granularity
                else:
                    st.warning("No 'Date' column found in the uploaded KPI data")
                    st.session_state['date_column'] = None
            else:
                st.info("Please upload KPI data to configure date granularity")
    
    with st.expander("**Demographic Data Selection**", expanded=True):
        # Determine which default columns to show based on market level
        if market_level == "US DMA":
            default_columns = DEFAULT_DMA_COLS
        elif market_level == "US State":
            default_columns = DEFAULT_STATE_COLS
        elif market_level == "BR Municipality":
            default_columns = DEFAULT_BR_COLS
        elif market_level == "MX Municipality":
            default_columns = DEFAULT_MX_COLS
        else:
            default_columns = DEFAULT_WORLD_COLS
            
        # Select demographic variables
        cov_columns = st.multiselect(
            "**Select Demographic Variables**",
            options=default_columns,
            default=default_columns[:4] if len(default_columns) > 4 else default_columns,
            help="Choose demographic variables to include in your analysis."
        )
        
        st.session_state['cov_columns'] = cov_columns
    
    # Process data and run analysis
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Run market ranking button
        run_market_ranking = st.button(
            "**Confirm and Run Market Ranking 🏃‍➡**",
            help="Start the market ranking analysis using the configured parameters."
        )
        
        if run_market_ranking:
            if ('kpi_df' in st.session_state and st.session_state['kpi_df'] is not None and 
                st.session_state['kpi_column'] is not None and 
                st.session_state['audience_column']):
                
                with st.spinner("Running Market Ranking Analysis..."):
                    try:
                        # Process data (placeholder for actual data processing)
                        # In a real implementation, this would involve fetching demographic data,
                        # merging with KPI data, preparing for analysis, etc.
                        
                        # For this example, we'll create a simple DataFrame to simulate the process
                        # This would be replaced with actual data processing in the complete implementation
                        kpi_df = st.session_state['kpi_df']
                        
                        # Sample market data based on selected market level
                        if st.session_state['market_level'] == "US DMA":
                            # Create sample DMA data with random values for demo
                            markets = kpi_df['Market'].unique()
                            df = pd.DataFrame({
                                'DMA Code': markets,
                                'DMA Name': [f"DMA {m}" for m in markets],
                                'KPI Tier': np.random.choice(['High', 'Medium', 'Low'], size=len(markets)),
                            })
                            
                            # Add demographic columns
                            for col in st.session_state['cov_columns']:
                                df[col] = np.random.normal(100000, 50000, size=len(markets))
                                
                            # Add audience columns
                            for col in st.session_state['audience_column']:
                                df[col] = np.random.normal(500000, 150000, size=len(markets))
                            
                            # Add client columns if available
                            if 'client_columns' in st.session_state and st.session_state['client_columns']:
                                for col in st.session_state['client_columns']:
                                    df[col] = np.random.normal(75000, 25000, size=len(markets))
                            
                            st.session_state['df'] = df
                            st.session_state['spend_cols'] = ["CLIENT_Media_Spend"] if "CLIENT_Media_Spend" in df.columns else []
                            
                            # Run matched market scoring
                            mm = MatchedMarketScoring(
                                df=df,
                                kpi_df=kpi_df,
                                audience_columns=st.session_state['audience_column'],
                                client_columns=st.session_state['client_columns'],
                                display_columns=[st.session_state['market_code'], st.session_state['market_name']],
                                covariate_columns=st.session_state['cov_columns'],
                                market_column=st.session_state['market_code'],
                                date_granularity=st.session_state['date_granularity'],
                                kpi_column=st.session_state['kpi_column'],
                                scoring_removed_columns=st.session_state['spend_cols'],
                                run_model=True
                            )
                            
                            st.session_state['mm'] = mm
                            st.session_state['mm1'] = mm
                            st.session_state['feature_importance'] = mm.feature_importance
                            
                            st.success("Market Ranking Analysis Completed Successfully!")
                            st.info("Please go to the 'Market Rankings & Insights' tab to view results.")
                        else:
                            # Similar process for other market levels
                            st.warning("This demo focuses on US DMA implementation. Processing for other market levels would follow similar patterns.")
                    except Exception as e:
                        st.error(f"Error running market ranking analysis: {str(e)}")
            else:
                st.error("Please upload KPI data, select a KPI column, and choose target audiences before running analysis.", icon="🚨")
