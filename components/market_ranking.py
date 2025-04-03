import json
import streamlit as st
import plotly.express as px
from scripts.constants import *
import streamlit_vertical_slider as svs
from scripts.matched_market import MatchedMarketScoring

def render_market_ranking():
    """
    This function renders the market ranking tab in the Streamlit app. It retrieves the session state,
    allows the user to customize factor weights, filters tiers, and visualizes market ranking and heatmaps.
    """

    # Extract required session state values
    mm, audience_columns, kpi_column, market_level, client_columns, \
    cov_columns, market_code, market_name, spend_cols, df, kpi_df, date_granularity = (
        st.session_state[key] for key in [
            "mm", "audience_column", "kpi_column", "market_level", 'client_columns',
            "cov_columns", "market_code", "market_name", "spend_cols", "df", "kpi_df", "date_granularity"
        ]
    )

    feature_importance = mm.feature_importance.copy()  # Copy feature importance
    replaced_values = {}  # Dictionary to store updated feature weights

    # Create layout for input columns
    col1, col2, col3, col4, col5 = st.columns(
        [0.15, 0.15, 0.30, 0.10, 0.10], gap="small"
    )

    # User input for enabling advanced features
    with col1:
        advanced_feature = st.selectbox(
            label="**Advanced Features Enabled**",
            options=["No", "Yes"],
            placeholder="No",
            help="Enable advanced features to customize factor weights.",
        )

    # User input for selecting top markets per tier
    with col2:
        top_n = st.number_input(
            label="**Top Markets Per Tier**",
            min_value=1,
            max_value=10,
            value=4,
            help="Choose the number of top markets you want displayed per tier.",
        )

    # User input for tier filtering
    with col3:
        tier_filter = st.multiselect(
            label="**Tier Filter**",
            options=set(mm.ranking_df[TIER]),
            default=set(mm.ranking_df[TIER]),
            help="Choose which tiers to include or exclude from the display.",
        )

    # If advanced features are enabled, adjust factor weights
    if advanced_feature == "Yes":
        with st.expander("**Advanced Features: Factor Weights Adjusting**", expanded=False):
            sorted_dict = sorted(
                mm.feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            top_10_keys = [item[0] for item in sorted_dict[:10]]
            columns = st.columns(len(top_10_keys))

            for i, c in enumerate(top_10_keys):
                locals()[f"col_{c}"] = columns[i]
                with locals()[f"col_{c}"]:
                    replaced_values[c] = svs.vertical_slider(
                        label=f"{c} Weight",  # Optional
                        height=200,  # Optional - Defaults to 300
                        thumb_shape="square",  # Optional - Defaults to "circle"
                        step=0.01,  # Optional - Defaults to 1
                        default_value=round(feature_importance.get(c), 2),
                        min_value=0,  # Defaults to 0
                        max_value=1,  # Defaults to 10
                        track_color="blue",  # Optional - Defaults to Streamlit Red
                        slider_color=("red", "blue"),  # Optional
                        thumb_color="orange",  # Optional - Defaults to Streamlit Red
                        value_always_visible=True,  # Optional - Defaults to False
                    )

            # Update feature importance with user-adjusted values
            for k, v in replaced_values.items():
                if v is not None:
                    feature_importance[k] = v

    # Normalize feature importance weights
    total_weight = sum([v for k, v in feature_importance.items()])
    feature_importance = {k: v / total_weight for k, v in feature_importance.items()}

    # Create a new instance of MatchedMarketScoring with updated parameters
    mm1 = MatchedMarketScoring(
        df=df,
        kpi_df=kpi_df,
        audience_columns=audience_columns,
        client_columns=client_columns,
        display_columns=[market_code, market_name],
        covariate_columns=cov_columns,
        market_column=market_code,
        date_granularity=date_granularity,
        kpi_column=kpi_column,
        run_model=False,
        feature_importance=feature_importance,
        scoring_removed_columns=spend_cols
    )

    # Saving to session state.
    st.session_state.mm1 = mm1
    st.session_state.feature_importance = feature_importance

    display_df1 = mm1.fi[[FEATURE, WEIGHT]]  # Dataframe for feature importance
    display_df1[WEIGHT] = display_df1[WEIGHT] / display_df1[WEIGHT].sum()  # Normalize weights
    display_df2 = mm1.ranking_df.reset_index(drop=True)  # Dataframe for market rankings

    st.write("")  # Empty line for spacing

    # If the market level is DMA, display the heat map
    if market_level == "Dma":
        with st.expander("**DMA Market Score Heat Map**", expanded=True):
            with open("dma.json") as geofile:
                source_geo = json.load(geofile)  # Load geographical data for the map

            # Prepare performance dataframe for heatmap
            perf_df = mm1.ranking_df.reset_index(drop=True)
            perf_df = perf_df[perf_df[TIER].isin(tier_filter)]

            # Create heat map using Plotly
            fig_heat = px.choropleth(
                perf_df,
                geojson=source_geo,
                locations='Market',
                color="Score",
                color_continuous_scale="YlOrRd",
                range_color=(0, perf_df["Score"].max()),
                scope="usa",
                labels={"Score": "Market Score"},
                hover_data=[market_code],
                height=1000,
                width=1200,
                title=" <span style='font-size:30px;color:black;'>DMA Market Score Heat Map </span>",
            )
            fig_heat.update_layout(title_x=0.375, title_y=0.9)  # Adjust title position
            st.plotly_chart(fig_heat, theme="streamlit", use_container_width=True)  # Render heat map

    # Expander for socioeconomic weights and market rankings
    with st.expander(f"**Socioeconomic Weights for {kpi_column} & Market Rankings**", expanded=True):
        col1, col2, col3 = st.columns([0.2, 1.5, 1.75])  # Layout for plots

        with col2:
            # Create pie chart for feature weights
            fig_feature_weights = px.pie(
                display_df1,
                values=WEIGHT,
                names=FEATURE,
                color=FEATURE,
                title=f"Socioeconomic Feature Weights for {kpi_column} Tiers",
                width=600,
                height=500,
            )
            st.plotly_chart(fig_feature_weights, theme="streamlit", use_container_width=False)  # Render pie chart

        with col3:
            # Filter the rankings dataframe based on user selections
            display_df2 = display_df2[
                (display_df2[TIER].isin(tier_filter)) &
                (display_df2["Tier Rank"] <= top_n)
            ].sort_values(by=[TIER, "Score"], ascending=[False, True])

            # Determine graph column based on market level
            # graph_col = (DMA_NAME if market_level == "DMA" else
            #              STATE_NAME if market_level == "State" else
            #              COUNTRY_NAME)

            # Create horizontal bar chart for top markets
            fig_ranking = px.bar(
                display_df2,
                x=SCORE,
                y=market_name,
                orientation="h",  # Horizontal bar chart
                labels={market_name: "Market Rank", SCORE: "Market Score"},
                title=f"Top {top_n} Markets Per Tier Based on {kpi_column} Score",
                width=600,
                height=500,
                color=TIER,
            )
            st.plotly_chart(fig_ranking, theme="streamlit")  # Render bar chart

        st.write("")  # Empty line for spacing

        # Display data frame of top markets
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown(
                f"<h5 style='text-align: center; color: black;'>Top {top_n} "
                f"Markets Per Tier Based on {kpi_column} Score</h5>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                display_df2[
                    [TIER, market_name, market_code, "Score", "Tier Rank"]
                ].sort_values(by=[TIER, "Score"], ascending=[True, False]),
                hide_index=True,
                use_container_width=True,
            )
