import streamlit as st
import pandas as pd

def render_product_quick_start():
    with st.expander("**✅ Steps to Run the Tool**", expanded=True):
        st.markdown("""
            Follow these steps to successfully run the tool:

            1. Go to the 'Matched Market Command Center' tab to start.
            2. First select the country and market level of your data.
            3. Choose your target audiences, up to three.         
            4. Ensure that your data adheres to the required schemas. You must upload client KPI data and optionally extra client specific data.
            5. Use the 'Data Uploader' to upload your files.             
            6. Configure the necessary KPI selections and the Date granularity for your analysis.
            7. Select additional data sources to enrich your dataset with demographic information.
            8. Execute the analysis by clicking the 'Confirm and Run Market Ranking' button.
            9. Review the results displayed in the 'Market Ranking & Insights' tab.
            10. Finally, in the 'Matched Markets' tab, you will find insights that enhance targeting strategies by identifying patterns and similarities between markets.
        """)

    with st.expander("📋 Atlas Data Requirements"):
        st.markdown("""
            You will need to upload the necessary data in the 'Matched Market Command Center' tab. Ensure that the databases being uploaded comply with the expected schema to guarantee smooth operation. The tool can be used only for USA (State and DMA level), Brazil (Municipality level), and Mexico (Municipality level).

            #### **Client KPI Data**
            **Columns**
            - **Market**
              - **_US DMA CODE_**
                - **Description:** A unique identifier for a designated market area, used to link audience data by geographic region.
                - **Naming Convention:** Market.
                - **Type:** Numeric.
                - **Format:** 3 digits (e.g., 500, 501, 503).
              - **_US STATE CODE_**
                - **Description:** A two-letter postal abbreviation for a U.S. state, used to associate audience data with specific states.
                - **Naming Convention:** Market.                
                - **Type:** Alphanumeric.
                - **Format:** 2 characters (e.g., NY, FL, TX).
              - **_BR Minicipality CODE_**
                - **Description:** A unique identifier of BR municipalities.
                - **Naming Convention:** Market.                
                - **Type:** Numeric.
                - **Format:** 7 digits (e.g., 1200013, 1200385, 2700201).
              - **_MX Municipality CODE_**
                - **Description:** A unique identifier combining the code of MX State + MX Municipality.
                - **Naming Convention:** Market.                
                - **Type:** Numeric.
                - **Format:** up to 5 digits (e.g., 12, 1205, 1343).
            - **Date**
              - **Description:** The date for the KPI data. Ideally, the data should be provided daily, but weekly data is acceptable as a minimum. If it is weekly data, this should be the start or end date of the week. 
              - **Naming Convention:** Date.
              - **Format:** _mm/dd/yyyy_ (this format is important and must be followed).
            - **KPIs**
              - **Description:** There should be one or more columns containing Key Performance Indicators (KPIs).
              - **Naming Convention:** _KPI_XXX_ (e.g., KPI_Signups, KPI_Total_Revenue). Each KPI column must include the word "KPI_" in its name.

            **Examples daily and weekly KPI Data:**
            """)

        # Create a DataFrame for KPI data example
        kpi_data = pd.DataFrame({
            "Market": ["FL", "GA", "HI", "ID"],
            "Date": ["7/28/24", "7/28/24", "7/28/24", "7/28/24"],
            "KPI_Users": [208, 129, 10, 7],
            "KPI_Engaged_Sessions": [101, 75, 7, 4]
        })
        st.dataframe(kpi_data, use_container_width=False, hide_index=True)
        
        # Create a DataFrame for weekly revenue example
        weekly_revenue_data = pd.DataFrame({
            "Market": [500, 501, 502, 503],
            "Date": ["10/2/21", "10/2/21", "10/2/21", "10/2/21"],
            "KPI_Total_Revenue": [60862.7, 2320011.25, 10095.65, 27612.6]
        })
        st.dataframe(weekly_revenue_data, use_container_width=False, hide_index=True)

        st.markdown("""
            #### **Optional Client Specific Data**
            **Columns**
            - **Market**
              - **_US DMA CODE_**
                - **Description:** A unique identifier for a designated market area, used to link audience data by geographic region.
                - **Naming Convention:** Market.
                - **Type:** Numeric
                - **Format:** 3 digits (e.g., 500, 501, 503)
              - **_US STATE CODE_**
                - **Description:** A two-letter postal abbreviation for a U.S. state, used to associate audience data with specific states.
                - **Naming Convention:** Market.
                - **Type:** Alphanumeric
                - **Format:** 2 characters (e.g., NY, FL, TX)
              - **_BR Minicipality CODE_**
                - **Description:** A unique identifier of BR municipalities.
                - **Naming Convention:** Market.                
                - **Type:** Numeric.
                - **Format:** 7 digits (e.g., 1200013, 1200385, 2700201).
              - **_MX Municipality CODE_**
                - **Description:** A unique identifier combining the code of MX State + MX Municipality.
                - **Naming Convention:** Market.                
                - **Type:** Numeric.
                - **Format:** up to 5 digits (e.g., 12, 1205, 1343).

            - **CLIENTs**
              - **Description:** There should be one or more columns containing client information such as media spend.      
              - **Naming Convention:** _CLIENT_XXX_ (e.g., CLIENT_Media_Spend, CLIENT_Total_Clicks). Each Client Data column must include the word "CLIENT_" in its name.
                         
            **Examples Client Specific Data:**
            """)
        
        # Create a DataFrame for audience data example
        audience_data = pd.DataFrame({
            "Market": ["FL", "GA", "HI", "ID"],
            "CLIENT_Media_Spend": [34555, 334565, 22345, 23443],
            #"AUDIENCE_F35+": [66998, 3456602, 200045, 160320]
        })
        st.dataframe(audience_data, use_container_width=False, hide_index=True)

        
        # Create a DataFrame for audience metrics example
        audience_metrics_data = pd.DataFrame({
            "Market": [500, 501, 502, 503],
            "CLIENT_Media_Spend": [98979, 2457307, 36390, 75838],
            #"AUDIENCE_F35+": [323977, 6255895, 100076, 190420]
        })
        st.dataframe(audience_metrics_data, use_container_width=False, hide_index=True)