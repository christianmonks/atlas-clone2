import streamlit as st
from scripts.matched_market import MatchedMarketScoring

def render_power_analysis():

    df, kpi_df, audience_columns, client_columns, market_code, market_name, \
    date_granularity, cov_columns, kpi_column, feature_importance, spend_cols = (
        st.session_state[key] for key in [
            "df", "kpi_df", "audience_column", "client_columns", "market_code", 'market_name',
            "date_granularity", "cov_columns", "kpi_column", "feature_importance", "spend_cols"
        ]
    )

    col1, col2, col3, col4, col5 = st.columns([0.3, 0.3, 0.3, 0.3, 0.3], gap="small")
    with col1:
        budget = st.number_input(
            "**Budget**",
            min_value=0,
            max_value=1000000,
            value=0,
            help="Total budget allocated for the test.",
        )
    with col2:
            cost = st.number_input(
                "**CPIK**",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                help="Cost per incremental unit of KPI.",
            )
    with col3:
        alpha = st.number_input(
            "**Significance Level**",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="Significance level (alpha). This determines the threshold for statistical significance.",
        )
    with col4:
        power = st.number_input(
            "**Power Level**",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            help="Statistical power level. This is the probability of correctly rejecting a false null hypothesis.",
        )
    with col5:
        lift = st.selectbox(
            "**Minimum Detectable lift**",
            options=['15%', '10%', '5%'],
            help="Minimum lift that can be detected by the test.",
        )

    st.write("")
    st.write("")
    bt_run_power_analysis = st.button(label="**Confirm and Run Power Analysis 🏃‍➡**")
    if bt_run_power_analysis:
        with st.spinner(
            text="Running Power Analysis..."
        ):
            mm2 = MatchedMarketScoring(
                df=df,
                kpi_df=kpi_df,
                audience_columns=audience_columns,
                client_columns=client_columns,
                display_columns=[market_code, market_name],
                covariate_columns=cov_columns,
                market_column=market_code,
                date_granularity=date_granularity,
                kpi_column=kpi_column,
                feature_importance=feature_importance,
                scoring_removed_columns=spend_cols,
                power_analysis_parameters={
                    'Alpha': alpha,
                    'Power': power,
                    'Lifts': [int(lift.replace('%', ''))]
                },
                power_analysis_inputs={
                    'Cost': cost,
                    'Budget': budget
                },
                run_model=False
            )

            # Display options
            if len(mm2.power_analysis_results.get('By Duration')) > 0:
                st.dataframe(
                    mm2.power_analysis_results.get('By Duration'),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.error("No feasible solution found. Considering increasing the budget.", icon="🚨")
