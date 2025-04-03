import numpy as np
import pandas as pd
import math
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scripts.constants import *
from statsmodels.stats.power import TTestIndPower

class MatchedMarketScoring:
    def __init__(
        self,
        df: pd.DataFrame,
        kpi_df: pd.DataFrame,
        client_columns: list,
        audience_columns: list,
        covariate_columns: list,
        kpi_column: str,
        display_columns: list = [DMA_CODE, DMA_NAME],
        market_column: str = DMA_CODE,
        date_granularity = 'Daily',
        target_variable: str = TIER,
        scoring_removed_columns: list = [],
        power_analysis_parameters: dict = {
            'Alpha': 0.1,
            'Power': 0.8,
            'Lifts': [15],
        },
        power_analysis_inputs: dict = {
            'Cost': None,
            'Budget': None,
        },
        run_model: bool = True,
        feature_importance: dict = None,
    ):
        """
        Initialize the MatchedMarketScoring class.

        :param df: DataFrame containing the aggregated data.
        :param kpi_df: DataFrame containing the raw data.
        :param client_columns: List of columns representing client-specific features.
        :param audience_columns: List of columns representing audience features.
        :param covariate_columns: List of columns representing covariates or control variables.
        :param display_columns: List of columns to display in the output (default: [DMA_CODE, DMA_NAME]).
        :param market_column: Column used to define the market (default: DMA_CODE).
        :param date_granularity: Date granularity for KPI (default: Daily).
        :param target_variable: Name of the target variable column (default: TIER).
        :param scoring_removed_columns: List of columns to exclude from scoring.
        :param power_analysis_parameters: Dictionary containing parameters for power analysis (alpha, power, lifts).
        :param power_analysis_inputs: Dictionary containing inputs for power analysis (budget, cost).
        :param run_model: Boolean indicating whether to run the model or use pre-calculated feature importance.
        :param feature_importance: Pre-calculated feature importance dictionary (default: None).
        """

        # Create a deep copy of the input DataFrame
        self.df = df.copy(deep=True)
        self.kpi_df = kpi_df.copy(deep=True)

        # Store input parameters
        self.target_variable = target_variable
        self.display_columns = display_columns
        self.covariate_columns = covariate_columns
        self.audience_columns = audience_columns
        self.client_columns = client_columns
        self.market_column = market_column
        self.date_granularity = date_granularity
        self.kpi_column = kpi_column

        # Combine model-related columns and filter them by what exists in the DataFrame
        self.model_columns = self.covariate_columns + self.audience_columns + self.client_columns
        self.model_columns = [i for i in self.model_columns if i in list(self.df)]

        # Handle columns to be removed from scoring
        self.scoring_removed_columns = scoring_removed_columns
        if self.scoring_removed_columns:
            for c in self.scoring_removed_columns:
                print(f"-------- Column: {c} Removed from Market Scoring Purposes --------")

        # If run_model is True, calculate feature importance, otherwise import existing one
        if run_model:
            print("-------- Calculating Feature Importance --------")
            self.feature_importance = self.run_model()
        else:
            print("-------- Importing Feature Importance --------")
            self.feature_importance = feature_importance
            # Check for missing feature importance for any of the model columns
            missing = [key for key in self.model_columns if key not in self.feature_importance]
            for i in missing:
                print(f"-------- Missing Feature Importance: {i} --------")

        # Extract feature importance values for ranking and scoring
        self.feature_weights = [v for k, v in self.feature_importance.items()]

        # Score markets based on the calculated or provided feature importance
        self.ranking_df, self.fi = self.score_markets()

        # Identify similar markets using the feature importance
        self.similar_markets = self.matching_markets()

        # Store power analysis inputs (budget, cost) and check for completeness before running power analysis
        self.power_analysis_inputs = power_analysis_inputs
        if self.power_analysis_inputs.get('Budget') and self.power_analysis_inputs.get('Cost'):
            print('-------- Running Power Analysis --------')
            for k, v in self.power_analysis_inputs.items():
                print(f'-------- {k}: ${v} --------')

            self.power_analysis_parameters = power_analysis_parameters
            # self.power_analysis_parameters['Lifts'] = [x / 100 for x in self.power_analysis_parameters['Lifts']]
            for k, v in self.power_analysis_parameters.items():
                print(f"-------- Power Analysis Parameter {k}: {v}  --------")
            self.power_analysis_results = self.power_analysis()
        else:
            print('-------- Please Input a Budget and a Cost to Run Power Analysis ---------')

    def run_model(self):
        """
        Perform model execution.

        This method executes the scoring process using a RandomForestClassifier model. It performs grid search
        for hyperparameter tuning, fits the best model, and returns the feature importances.

        :return: A dictionary representing feature importances.
        """
        df = self.df.dropna(subset=self.model_columns).reset_index(drop=True)
        x = df[self.model_columns]
        y = df[self.target_variable]

        param_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "max_features": ["sqrt", None],
            "n_estimators": [10, 20, 50, 100, 200],
        }

        model = RandomForestClassifier(random_state=100)
        grid_search = GridSearchCV(
            estimator=model,
            scoring="accuracy",
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid_search.fit(x, y)
        best_model = grid_search.best_estimator_
        feature_importance = {
            list(x.columns)[i]: v for i, v in enumerate(best_model.feature_importances_)
        }
        return feature_importance

    def score_markets(self):
        """
        Evaluate market scores.

        This method calculates scores for markets based on feature importances, applies scaling, and computes overall and
        tier rankings. It returns DataFrames containing market rankings and feature importances.

        :return: Tuple comprising a DataFrame for market rankings and a DataFrame for feature importances.
        """
        df = self.df.dropna(subset=self.model_columns).reset_index(drop=True)
        fi = pd.DataFrame(
            list(self.feature_importance.items()), columns=["Feature", WEIGHT]
        ).sort_values(by=[WEIGHT], ascending=False)
        score_df = df[self.model_columns].copy()
        score_df = score_df[[c for c in list(score_df) if c not in self.scoring_removed_columns]]

        scaler = MinMaxScaler()
        x_norm = scaler.fit_transform(score_df)
        total_feat = sum([v for c,v in self.feature_importance.items() if c in list(score_df)])
        scores = np.matmul(x_norm, [v/total_feat for c,v in self.feature_importance.items() if c in list(score_df)])

        ranking_df = pd.DataFrame(x_norm, columns=list(score_df))
        ranking_df[SCORE] = list(scores)

        display_columns = self.display_columns + [self.target_variable]
        ranking_df = pd.concat([df[display_columns], ranking_df], axis=1)
        ranking_df = ranking_df.sort_values(by=[SCORE], ascending=False).reset_index(
            drop=True
        )
        ranking_df["Overall Rank"] = ranking_df[SCORE].rank(ascending=False).astype(int)
        ranking_df["Tier Rank"] = (
            ranking_df.groupby([TIER])[SCORE].rank(ascending=False).astype(int)
        )
        return ranking_df, fi

    def matching_markets(self):
        """
        Find similar markets based on ranking DataFrame.

        :return: DataFrame containing similar markets information.
        """
        similar_markets = pd.DataFrame()
        for t in set(self.ranking_df[self.target_variable]):
            market_tier = self.ranking_df[
                self.ranking_df[self.target_variable] == t
            ].reset_index(drop=True)
            market_dist = market_tier[
                [c for c in self.model_columns if c not in self.scoring_removed_columns]
            ]

            for i, v in market_dist.iterrows():
                rank = market_dist.apply(lambda row: abs(v - row), axis=1)
                markets = pd.DataFrame(
                    {
                        "KPI Tier": [t] * len(market_dist),
                        "Test Market Identifier": [
                            market_tier[f"{self.market_column}"].iloc[i]
                        ]
                        * len(market_dist),
                        "Test Market Name": [
                            market_tier[
                                f"{self.display_columns[1]}"
                            ].iloc[i]
                        ]
                        * len(market_dist),
                        "Control Market Identifier": [
                            i for i in market_tier[f"{self.market_column}"]
                        ],
                        "Control Market Name": [
                            i
                            for i in market_tier[
                                f"{self.display_columns[1]}"
                            ]
                        ],
                        "Similarity Index": list(
                            np.matmul(rank, [
                                v for c,v in self.feature_importance.items() if c not in self.scoring_removed_columns
                            ])
                        ),
                        "Test Market Score": [market_tier[SCORE].iloc[i]]
                        * len(market_dist),
                        "Control Market Score": [i for i in market_tier[SCORE]],
                    }
                )
                similar_markets = pd.concat([similar_markets, markets], axis=0)
        similar_markets = similar_markets[
            similar_markets["Test Market Identifier"]
            != similar_markets["Control Market Identifier"]
        ]

        similar_markets["Similarity Index"] = 1 - similar_markets["Similarity Index"]
        return similar_markets

    def power_analysis(self):
        """
        Perform power analysis to calculate the required sample size, budget,
        and running time for each market and lift percentage.
        """

        # Extract the relevant data (Market and KPI column) from the dataframe
        kpi_data = self.kpi_df[['Market', self.kpi_column]]

        # Extract power analysis input: budget limit and cpik
        budget_limit = self.power_analysis_inputs.get('Budget')
        cpik = self.power_analysis_inputs.get('Cost')

        # Extract the list of similar markets (Test Market and Control Market pairs)
        matched_markets = st.session_state['matched_markets']
        markets = matched_markets[['Test Market Identifier', 'Control Market Identifier']]
        l_markets = markets.values.tolist()
        test_market = matched_markets['Test Market Identifier'].values.tolist()
        control_market = matched_markets['Control Market Identifier'].values.tolist()

        testing_markets, results = [], []
        # Iterate over each market pair
        for market in l_markets:
            # Append the current market to the testing markets list
            testing_markets.append(market)

            # Flatten the list of testing markets
            geo_input = [item for sublist in testing_markets for item in sublist]

            # Filter the KPI data for the selected markets
            geo_data = kpi_data[kpi_data['Market'].isin(geo_input)]

            # Calculate the mean and standard deviation for the KPI
            kpi_mean = geo_data[self.kpi_column].mean()
            kpi_std = geo_data[self.kpi_column].std()

            # Perform power analysis for each lift percentage
            for lift in self.power_analysis_parameters.get('Lifts'):
                # Calculate the minimum lift as a percentage of the mean KPI
                min_lift_percentage = lift / 100

                # Calculate the effect size for the t-test
                effect_size = min_lift_percentage * kpi_mean / kpi_std

                # Initialize power analysis using a t-test for independent samples
                analysis = TTestIndPower()

                # Solve for the required sample size given the effect size, power, and alpha
                obs = analysis.solve_power(
                    effect_size,
                    power=self.power_analysis_parameters.get('Power'),
                    alpha=self.power_analysis_parameters.get('Alpha'),
                    ratio=1, #ratio of test to control markets
                    alternative='two-sided',
                )

                # Calculate the required budget based on sample size, KPI mean, and lift
                budget = obs * kpi_mean * min_lift_percentage * cpik

                # Estimate the running time in weeks (proportional to sample size)
                running_time = 2 * obs / len(geo_input)

                # Append the results for the current lift to the results list
                # Date granularity can be Daily or Weekly
                if self.date_granularity == 'Weekly':
                    results.append({
                        'Number of Test Markets': int(len(geo_input) / 2),
                        'Budget': int(budget),
                        'Running Time (Weeks)': math.ceil(running_time),
                        'Test Markets': ','.join(map(str, [x for x in geo_input if x in test_market])),
                        'Control Markets': ','.join(map(str, [x for x in geo_input if x in control_market]))
                    })
                elif self.date_granularity == 'Daily':
                    results.append({
                        'Number of Test Markets': int(len(geo_input) / 2),
                        'Budget': int(budget),
                        'Running Time (Weeks)': math.ceil(running_time / 7),
                        'Test Markets': ','.join(map(str, [x for x in geo_input if x in test_market])),
                        'Control Markets': ','.join(map(str, [x for x in geo_input if x in control_market]))
                    })

        # Convert the results list into a pandas DataFrame
        df_results = pd.DataFrame(results)
        filtered_rows = df_results[(df_results['Budget'] < budget_limit)] \
            .sort_values(by=['Running Time (Weeks)', 'Budget'])

        result_dict = {
            'All Results': df_results,
            'By Duration': filtered_rows,
        }
        return result_dict

# def generate_dma_data(
#         dma_census_data: dict,
#         included_datasets=[],
# ):
#     """
#     Generate DMA output based on census data.
#
#     :return: DataFrame containing DMA output
#     """
#     dma_data = copy.deepcopy(
#         dma_census_data
#     )
#
#     dataframes = {
#         k: v for k, v in dma_data.items() if k != 'CPM DMA'
#     }
#
#     if len(included_datasets) > 0:
#         dataframes = {
#             k: v for k, v in dataframes.items() if k in included_datasets
#         }
#
#     base = dma_data.get('CPM DMA')
#     for n, df in dataframes.items():
#         print(f'Adding in DataFrame: {n}')
#         nc = n.replace("DMA", "").strip()
#         if 'Median' not in n:
#             df.columns = [f"{k}_{nc.lower().replace(' ','_')}" if k not in [
#                 DMA_CODE, DMA_NAME
#             ] else k for k in list(df)]
#         else:
#             df = df.rename(columns={'median': nc})
#         base = base.merge(
#             df, on=[DMA_CODE, DMA_NAME], how='left'
#         )
#     return base

def calculate_tier(pct_rank, num_tiers):
    interval = 1 / num_tiers
    thresholds = [interval * i for i in range(1, num_tiers)]
    tiers = [f"Tier {i}" for i in range(num_tiers, 0, -1)]

    for threshold, tier in zip(thresholds, tiers):
        if pct_rank <= threshold:
            return tier
    return tiers[-1]
