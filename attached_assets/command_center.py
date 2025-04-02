import pandas as pd
import streamlit as st
import os
import regex as re
from os.path import join
from pandas.api.types import is_numeric_dtype
from scripts.constants import *
from scripts.matched_market import calculate_tier, MatchedMarketScoring

def render_command_center():
    """
    Render the Command Center tab in the Streamlit app.
    """

    # Set the current directory
    # Initialize empty dataframes

    cd = os.getcwd()
    kpi_df, client_df, agg_kpi_df, audience_df, df = None, None, None, None, None
    client_columns, audience_columns = [], []

    # Expander for Market Level Selection
    with st.expander(label="**Market Level Selection**", expanded=True):

        market_level = st.selectbox(
            label="**Select a Market Level**",
            options= ['Select a Market Level'] + MARKET_LEVELS,
            help="Choose a Market Level for Market Scoring & Market Matching"
        )

        country_abbreviation = market_level.split()[0].upper() # US, BR, MX

        column_market_name = market_level.split()[-1].lower().capitalize() # DMA, State, Municipality 

        #print(column_market_name) 


    # Expander for Target data selection
    with st.expander(label="**Target Audience**", expanded=True):

        if market_level == 'Select a Market Level':
            st.error("Please Return to the Previous Expander and select a Market Level", icon="🚨")

        elif market_level == 'Other':
            st.error('This option is not available for "Other".', icon="⚠️")
        else:
            # Read audience csv by market and country
            audience_file = f"{market_level.replace(' ', '_').lower()}_audience.csv"
            audience_path = join(cd, 'data', 'audience', audience_file)
            complete_audience_df = pd.read_csv(audience_path, dtype={MARKET_COLUMN: str})

            
            order = ['Females', 'Males', 'Population', 'Females Baby Boomers','Males Baby Boomers','Population Baby Boomers','Females Millennials','Males Millennials','Population Millennials','Females Generation Z','Males Generation Z','Population Generation Z']

            
            f_columns = [col for col in complete_audience_df.columns if re.match(r'^F\d', col)]

            m_columns = [col for col in complete_audience_df.columns if re.match(r'^M\d', col)]

            p_columns = [col for col in complete_audience_df.columns if re.match(r'^P\d', col)]

            # Reorganizar DataFrame según el nuevo orden
            audience_columns = order + f_columns + m_columns + p_columns
            default = "Population"

            #select-box to multiselect for multiple selection
            audience_seletion = st.multiselect(
                label="**Select the Audiences for the Campaign**",  # Changed to plural
                options=audience_columns,
                default=[default] if default in audience_columns else [],  # Set the default value
                help="Select up to 3 audience demographics. You can select 1, 2, or 3.",  # Updated help message
                key="audience_selection"  # Keep the key to manage state if needed
            )

            audience_filter = None

            # Check if more than 3 audiences have been selected
            if len(audience_seletion) > 3:
                st.warning("You can select a maximum of 3 audiences.")  # Warning for exceeding the limit
                audience_filter = None # set as None
            else:
                # Generate a filtered DataFrame only if 1 to 3 audiences are selected
                if len(audience_seletion) > 0 and len(audience_seletion) < 4:
                    audience_filter = audience_seletion.copy()
                    audience_df = complete_audience_df[[MARKET_COLUMN] + audience_filter]  # Create filtered DataFrame

           
    # Expander for Data Upload
    with st.expander(label="**Data Uploader**", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            # KPI data uploader
            st.write("")
            uploaded_file_kpi = st.file_uploader("**Upload Client KPI Data**")
            if uploaded_file_kpi is not None:
                # Load and display KPI data if uploaded
                kpi_df = pd.read_csv(uploaded_file_kpi, dtype={MARKET_COLUMN: str})

                kpi_column_exists = any("kpi" in v.lower() for v in kpi_df.columns)
                if len(kpi_df) > 0 and kpi_column_exists:
                    st.success(
                        f"Successfully Loaded KPI Data. Total of {len(kpi_df)} Records. A Snapshot Is Provided Below."
                    )
                    st.dataframe(kpi_df, hide_index=True)


                else:
                    st.error("Please load a compatible dataset with a defined KPI column.")
            else:
                st.error("No Client KPI Data Uploaded", icon="🚨")

        with col2:
            st.write("")
            uploaded_file_client = st.file_uploader("**Upload Optional Client Specific Data**")
            if uploaded_file_client is not None:
                client_df = pd.read_csv(uploaded_file_client, dtype={MARKET_COLUMN: str})
                if len(client_df) > 0:
                    st.success(
                        f"Successfully Loaded Client Data. Total of {len(client_df)} Records. A Snapshot Is Provided Below."
                    )
                    st.dataframe(client_df, hide_index=True)
            else:
                st.error("No Additional Client Data Uploaded (Optional)", icon="🚨")

    # Expander for Data Granularity
    with st.expander(label="**KPI Selection and Data Granularity**", expanded=True):
        # Ensure KPI data is uploaded before proceeding
        if kpi_df is None:
            st.error("Please return to the previous expander and upload KPI data.", icon="🚨")
        else:
            kpi_columns = {
                i.replace("KPI", "").title().replace("_", " ").strip(): i
                for i in kpi_df.columns if "kpi" in i.lower()
            }
            kpi_column = st.selectbox(
                label="**Select a KPI Column to Rank Markets By**",
                options=list(kpi_columns.keys()),
                help="Choose a KPI column from the available options. Markets will be grouped into tiers based on the selected KPI."
            )
            rename_dict = {
                market_level: MARKET_COLUMN,
                kpi_columns.get(kpi_column): kpi_column,
            }

            # Rename columns in client dataframe if it exists
            if client_df is not None:
                client_columns = {i: i.title().replace("_", " ") for i in client_df.columns if market_level.lower() not in i.lower()}
                client_df.rename(columns=lambda col: rename_dict.get(col, col), inplace=True)
                client_df.rename(columns=lambda col: client_columns.get(col, col), inplace=True)
                client_columns = list(client_columns.values())
            
            # Rename columns in KPI dataframe
            kpi_df.rename(columns=lambda col: rename_dict.get(col, col), inplace=True)

            date_column = next((col for col in kpi_df.columns if col == 'Date' or col == 'date'), None)

            # Check if market level exists in KPI data
            if MARKET_COLUMN in kpi_df.columns:
                if is_numeric_dtype(kpi_df[kpi_column]):
                    num_tiers = st.number_input(
                        min_value=1,
                        max_value=10,
                        value=4,
                        label="**Number of KPI Tiers**",
                        help="For numeric KPI columns, choose the number of tiers to group markets by performance."
                    )
                    agg_kpi_df = kpi_df.groupby(MARKET_COLUMN)[[kpi_column]].sum().reset_index()
                    agg_kpi_df[PERCENT_RANK] = agg_kpi_df[kpi_column].rank(pct=True)
                    agg_kpi_df[TIER] = agg_kpi_df[PERCENT_RANK].apply(lambda x: calculate_tier(x, num_tiers))
                else:
                    # Non-numeric KPI - Copy as is
                    agg_kpi_df = kpi_df.copy()
                    agg_kpi_df[TIER] = agg_kpi_df[kpi_column]

                if date_column:
                    # Select a date column if available
                    date_column_granularity = st.selectbox(
                        label="**Select the granularity level of the date column**",
                        options=['Daily', 'Weekly'],
                        help="Select the level of aggregation of the date column of the KPI."
                    )
                else:
                    date_column_granularity = None
            else:
                st.error("Confirm Market Level", icon="🚨")


    
    # Expander for incorporating additional data sources
    with st.expander(label="**Incorporating Additional Data Sources**"):
        
        if (kpi_df is None) or (market_level is None) or (MARKET_COLUMN not in list(kpi_df)):
            st.error("Please Return to the Previous Expander and Upload Audience and KPI Data", icon="🚨")
        elif audience_filter is None and market_level != "Other":
            st.error("Please Return to the Target Audience Expander and select correctly the audiences, can select maximum 3 audiences.", icon="🚨")
        else:

            df = agg_kpi_df

            if market_level != "Other":
                # Load and process additional data
                additional_data = pd.read_csv(join(cd, 'data', 'census', f'{market_level.replace(" ", "_").lower()}_data.csv'), dtype={MARKET_COLUMN: str})
                additional_data = additional_data.rename(columns=lambda col: rename_dict.get(col, col))

                if audience_df is not None:
                    additional_columns_exclude = audience_columns
                    additional_data = additional_data[[k for k in list(additional_data) if k not in additional_columns_exclude]]

            # Start with the list of dataframes
            
                dfs_to_merge = [additional_data, client_df, audience_df]

                filter_columns_cov =  [MARKET_COLUMN, column_market_name, TIER, kpi_column, 'Percent Rank']

            else: 
                dfs_to_merge = [client_df]
                filter_columns_cov = [MARKET_COLUMN, TIER, kpi_column, 'Percent Rank']
                cols_additional_data = list(df.columns) - [kpi_column]
                additional_data = df[cols_additional_data]

            # Filter out None or empty dataframes
            # Perform merging if there are any valid dataframes
            dfs_to_merge = [df for df in dfs_to_merge if df is not None]

            print(df.columns)
            for additional_df in dfs_to_merge:
                print(additional_df.columns)

            for additional_df in dfs_to_merge:
                df = df.merge(additional_df, on=MARKET_COLUMN, how='inner')
            
            # Drop columns with high null percentage
            null_percentage = (df.isnull().sum() / len(df)) * 100
            columns_to_drop = null_percentage[null_percentage > 10].index
            df = df.drop(columns=columns_to_drop)

            cov_columns = [c for c in additional_data if c not in filter_columns_cov ]
            cov_columns = {c: c.title().replace("_", " ") for c in cov_columns}

            df = df.rename(columns=cov_columns)
            cov_columns = [v for k, v in cov_columns.items()]
            cov_columns = list(set(cov_columns))
            print(f'cov_columns: {cov_columns}')
            print(f' kpi : {kpi_column}')

            # Identify high correlation covariates if KPI is numeric
            if is_numeric_dtype(kpi_df[kpi_column]):
                corr = df[cov_columns + [kpi_column]].corr()[kpi_column].reset_index()
                print(f'corr: {corr}')
                corr_vars = [
                    i for i in corr[corr[kpi_column] > VARIABLE_CORRELATION_THRESHOLD]['index'].tolist() \
                    if i != kpi_column and i != 'Population'
                ]
                print(f'entre aca DEFAULT_COLUMNS:{corr_vars}')
                
                if corr_vars == [] or len(corr_vars) > MAX_DEFAULT:
                    default_columns = DEFAULT_COLUMNS.get(market_level.replace(' ', '_'))
                    corr_vars = default_columns

            else:
                # Default columns for non-numeric KPI
                default_columns = DEFAULT_COLUMNS.get(market_level.replace(' ', '_'))
                corr_vars = default_columns
                print('corr_vars')
                print(market_level.replace(' ', '_'))
                print(corr_vars)


            # Multiselect for demographic factors
            included_cov = st.multiselect(
                label="**Select Demographic Factors to Include or Exclude**",
                options=cov_columns,
                default=corr_vars,
                help="Select demographic factors to include/exclude from the analysis."
            )

            # Final list of columns for analysis
            client_columns = client_columns  if client_columns is not None else []
            client_columns = [col for col in client_columns if col != MARKET_COLUMN]
            final_columns = [MARKET_COLUMN, column_market_name] + client_columns + audience_filter + included_cov  + [kpi_column, TIER] 
            df = df[final_columns]

            if st.checkbox("View Merged KPI, Audiences and Market Data"):
                st.dataframe(df, hide_index=True)
            st.success(
                "Successfully Merged KPI, Audiences, and Market Data. Review the merged data below."
            )

    # Run market ranking and matching if data is ready
    # The model requires the following variables:
    # 1. df: A DataFrame containing all necessary data
    #    - Includes columns: Market, Dma/State, KPI_columns,
    #      client_columns (audiences uploaded by the user), 
    #      audience_columns (audience selected by the user), and KPI_TIER (the target variable).
    #
    # 2. audience_column: The audience selected by the user.
    #
    # 3. client_columns: A list of audiences uploaded by the user.
    #    'Market' should be included in df but not in client_columns.
    #
    # 4. cov_columns: Columns with additional information.
    #
    # 5. display_columns: Includes Market and market_name: Dma/State.
    #   
    #   The model:
    # - Takes the complete df provided.
    # - Combines the three lists: audience_columns + client_columns + cov_columns.
    # - Filters df based on this comprehensive list as input for the model.
    # - Filters df using the target variable KPI_TIER by default.
    # - Runs the model.
    # - Uses display_columns to assign the model results (Market and Dma/State).


    if df is not None:
        bt_run_market_ranking = st.button(
            label="**Confirm and Run Market Ranking 🏃‍➡**"
        )
        if bt_run_market_ranking:
            with st.spinner(
                text="Running ML Model to Calculate Market Scoring & Matching..."
            ):
                spend_cols = [c for c in list(df) if 'spend' in c.lower()]
                mm = MatchedMarketScoring(
                    df=df,
                    kpi_df=kpi_df,
                    client_columns=client_columns,
                    audience_columns=audience_filter,
                    display_columns=[MARKET_COLUMN, column_market_name],
                    covariate_columns=cov_columns,
                    kpi_column = kpi_column,
                    market_column=MARKET_COLUMN,
                    date_granularity=date_column_granularity,
                    scoring_removed_columns=spend_cols
                )

            print('market_level: ')
            print(market_level)
            print(column_market_name)

            # Save model outputs to session state
            saved_outputs = {
                'mm': mm,
                'df': df,
                'kpi_df': kpi_df,
                'audience_column': audience_filter,
                'client_columns': client_columns,
                'kpi_column': kpi_column,
                'market_level': column_market_name,
                'cov_columns': cov_columns,
                'date_granularity': date_column_granularity,
                'market_code': str(MARKET_COLUMN),
                'market_name': column_market_name,
                'spend_cols': spend_cols,
                'date_column': date_column
            }
            st.session_state.update(saved_outputs)
            st.success("🚀 Successfully Ran Market Scoring & Matching 🚀")

    # Footer markdown with contact info
    st.markdown("***")
    st.markdown(
        "If you have any questions about the KPI or Audience data required, please reach out to Media Analytics "
        "team at MediaAnalytics@mediamonks.com"
    )
