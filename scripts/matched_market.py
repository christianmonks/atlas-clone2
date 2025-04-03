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
        # Handle state level column names
if 'State' in self.df.columns:
    self.display_columns = [MARKET_COLUMN, 'State']
else:
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
            self.power_analysis_results = {'By Duration': pd.DataFrame()}

    def run_model(self):
        """
        Perform model execution.

        This method executes the scoring process using a RandomForestClassifier model. It performs grid search
        for hyperparameter tuning, fits the best model, and returns the feature importances.

        :return: A dictionary representing feature importances.
        """
        # Check if self.model_columns is empty
        if not self.model_columns:
            print("-------- No valid model columns found. Using default feature importance. --------")
            # Return a default feature importance if no columns are available
            return {"default_feature": 1.0}
        
        # Check if the DataFrame has enough rows for model training
        if len(self.df) < 5:  # Minimum for 5-fold cross-validation
            print("-------- Not enough data rows for model training. Using default feature importance. --------")
            # Return equal importance for all features
            return {col: 1.0/len(self.model_columns) for col in self.model_columns}
        
        # Drop rows with NA values in model columns
        df = self.df.dropna(subset=self.model_columns).reset_index(drop=True)
        
        # Check if we still have enough data after dropping NA values
        if len(df) < 5:
            print("-------- Not enough non-NA data for model training. Using default feature importance. --------")
            return {col: 1.0/len(self.model_columns) for col in self.model_columns}
        
        # Ensure the target variable exists and has at least two classes
        if self.target_variable not in df.columns:
            print(f"-------- Target variable {self.target_variable} not found. Using default feature importance. --------")
            return {col: 1.0/len(self.model_columns) for col in self.model_columns}
        
        # Check if target variable has at least two classes for classification
        if len(df[self.target_variable].unique()) < 2:
            print(f"-------- Target variable {self.target_variable} has fewer than 2 classes. Using default feature importance. --------")
            return {col: 1.0/len(self.model_columns) for col in self.model_columns}
        
        # Prepare the data
        x = df[self.model_columns]
        y = df[self.target_variable]
        
        # Check for empty feature matrix
        if x.empty or y.empty:
            print("-------- Empty feature matrix or target vector. Using default feature importance. --------")
            return {col: 1.0/len(self.model_columns) for col in self.model_columns}

        param_grid = {
            "criterion": ["gini", "entropy"],  # Removed "log_loss" which might cause issues
            "max_depth": [3, 5, 7, 10],  # Simplified param grid
            "max_features": ["sqrt"],  # Removed None which might cause issues
            "n_estimators": [50, 100],  # Simplified param grid
        }

        try:
            model = RandomForestClassifier(random_state=100)
            grid_search = GridSearchCV(
                estimator=model,
                scoring="accuracy",
                param_grid=param_grid,
                cv=min(5, len(df)),  # Ensure CV doesn't exceed sample size
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            grid_search.fit(x, y)
            best_model = grid_search.best_estimator_
        except Exception as e:
            print(f"-------- Model training error: {str(e)}. Using default feature importance. --------")
            return {col: 1.0/len(self.model_columns) for col in self.model_columns}
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
        # Handle the case with default feature importance
        if list(self.feature_importance.keys()) == ["default_feature"]:
            print("-------- Using default feature importance for scoring --------")
            # Create even feature importance for available model columns
            self.feature_importance = {col: 1.0/len(self.model_columns) for col in self.model_columns}
            
        df = self.df.dropna(subset=self.model_columns).reset_index(drop=True)
        fi = pd.DataFrame(
            list(self.feature_importance.items()), columns=["Feature", WEIGHT]
        ).sort_values(by=[WEIGHT], ascending=False)
        
        # Check if score_df can be created
        try:
            score_df = df[self.model_columns].copy()
            score_df = score_df[[c for c in list(score_df) if c not in self.scoring_removed_columns]]
            
            # Ensure we have data to score
            if score_df.empty:
                print("-------- Empty score dataframe. Using default scores. --------")
                ranking_df = df[self.display_columns + [self.target_variable]].copy()
                ranking_df[SCORE] = 0.5  # Default mid-range score
                ranking_df["Overall Rank"] = ranking_df.index + 1
                ranking_df["Tier Rank"] = ranking_df.groupby([TIER]).cumcount() + 1
                return ranking_df, fi
                
            scaler = MinMaxScaler()
            x_norm = scaler.fit_transform(score_df)
            
            # Get the intersection of feature importance keys and score_df columns
            valid_features = [c for c in self.feature_importance.keys() if c in list(score_df)]
            
            if not valid_features:
                print("-------- No valid features for scoring. Using default scores. --------")
                scores = np.ones(len(score_df)) * 0.5  # Default mid-range score
            else:
                total_feat = sum([v for c,v in self.feature_importance.items() if c in list(score_df)])
                
                if total_feat == 0:
                    print("-------- Zero total feature importance. Using equal weights. --------")
                    scores = np.ones(len(score_df)) * 0.5  # Default mid-range score
                else:
                    try:
                        # Try to calculate scores using matrix multiplication
                        scores = np.matmul(x_norm, [v/total_feat for c,v in self.feature_importance.items() if c in list(score_df)])
                    except Exception as e:
                        print(f"-------- Error calculating scores: {str(e)}. Using default scores. --------")
                        scores = np.ones(len(score_df)) * 0.5  # Default mid-range score
        except Exception as e:
            print(f"-------- Error in score_markets: {str(e)}. Using default scores. --------")
            ranking_df = df[self.display_columns + [self.target_variable]].copy()
            ranking_df[SCORE] = 0.5  # Default mid-range score
            ranking_df["Overall Rank"] = ranking_df.index + 1
            ranking_df["Tier Rank"] = ranking_df.groupby([TIER]).cumcount() + 1
            return ranking_df, fi

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
        try:
            similar_markets = pd.DataFrame()
            
            # Check if ranking_df exists and has the target variable column
            if not hasattr(self, 'ranking_df') or self.target_variable not in self.ranking_df.columns:
                print("-------- Missing ranking dataframe or target variable. Returning empty similar markets. --------")
                return similar_markets
                
            # Get unique values of the target variable
            tiers = set(self.ranking_df[self.target_variable])
            if not tiers:
                print("-------- No tiers found in target variable. Returning empty similar markets. --------")
                return similar_markets
            
            # Check if model_columns exist in ranking_df
            valid_model_cols = [c for c in self.model_columns if c in self.ranking_df.columns and c not in self.scoring_removed_columns]
            if not valid_model_cols:
                print("-------- No valid model columns for matching. Returning empty similar markets. --------")
                return similar_markets
                
            for t in tiers:
                try:
                    # Filter ranking_df for current tier
                    market_tier = self.ranking_df[
                        self.ranking_df[self.target_variable] == t
                    ].reset_index(drop=True)
                    
                    if market_tier.empty:
                        print(f"-------- No markets in tier {t}. Skipping. --------")
                        continue
                        
                    # Check if market column exists
                    if self.market_column not in market_tier.columns:
                        print(f"-------- Market column {self.market_column} not found. Skipping tier {t}. --------")
                        continue
                    
                    # Check if display column exists
                    if len(self.display_columns) < 2 or self.display_columns[1] not in market_tier.columns:
                        print(f"-------- Display name column not found. Skipping tier {t}. --------")
                        continue
                    
                    # Filter for valid model columns
                    market_dist = market_tier[valid_model_cols]
                    
                    if market_dist.empty:
                        print(f"-------- No valid feature data for tier {t}. Skipping. --------")
                        continue
                    
                    # Check feature importance
                    valid_features = [c for c in self.feature_importance.keys() if c in valid_model_cols]
                    if not valid_features:
                        print(f"-------- No valid feature importance for tier {t}. Skipping. --------")
                        continue
                    
                    for i, v in market_dist.iterrows():
                        try:
                            # Calculate feature differences between markets
                            rank = market_dist.apply(lambda row: abs(v - row), axis=1)
                            
                            # Calculate weighted similarity scores
                            try:
                                similarity_scores = np.matmul(rank, [
                                    self.feature_importance.get(c, 0) for c in valid_model_cols
                                ])
                            except Exception as e:
                                print(f"-------- Error calculating similarity scores: {str(e)}. Using default scores. --------")
                                similarity_scores = np.ones(len(market_dist)) * 0.5
                            
                            # Create DataFrame for current market
                            markets = pd.DataFrame(
                                {
                                    "KPI Tier": [t] * len(market_dist),
                                    "Test Market Identifier": [
                                        market_tier[f"{self.market_column}"].iloc[i]
                                    ] * len(market_dist),
                                    "Test Market Name": [
                                        market_tier[f"{self.display_columns[1]}"].iloc[i]
                                    ] * len(market_dist),
                                    "Control Market Identifier": [
                                        i for i in market_tier[f"{self.market_column}"]
                                    ],
                                    "Control Market Name": [
                                        i for i in market_tier[f"{self.display_columns[1]}"]
                                    ],
                                    "Similarity Index": list(similarity_scores),
                                    "Test Market Score": [market_tier[SCORE].iloc[i]] * len(market_dist),
                                    "Control Market Score": [i for i in market_tier[SCORE]],
                                }
                            )
                            similar_markets = pd.concat([similar_markets, markets], axis=0)
                        except Exception as e:
                            print(f"-------- Error processing market {i} in tier {t}: {str(e)}. Skipping. --------")
                            continue
                except Exception as e:
                    print(f"-------- Error processing tier {t}: {str(e)}. Skipping. --------")
                    continue
            
            # Remove self-matches
            if not similar_markets.empty:
                try:
                    similar_markets = similar_markets[
                        similar_markets["Test Market Identifier"] != similar_markets["Control Market Identifier"]
                    ]
                    
                    # Convert similarity to distance
                    similar_markets["Similarity Index"] = 1 - similar_markets["Similarity Index"]
                except Exception as e:
                    print(f"-------- Error in final processing: {str(e)}. --------")
            
            return similar_markets
            
        except Exception as e:
            print(f"-------- Error in matching_markets: {str(e)}. Returning empty dataframe. --------")
            return pd.DataFrame()

    def power_analysis(self):
        """
        Perform power analysis to calculate the required sample size, budget,
        and running time for each market and lift percentage.
        """
        try:
            # Check if required data and parameters exist
            if not hasattr(self, 'kpi_df') or self.kpi_df is None or self.kpi_df.empty:
                print("-------- KPI data not available for power analysis. --------")
                return {'By Duration': pd.DataFrame()}
                
            if self.kpi_column not in self.kpi_df.columns:
                print(f"-------- KPI column '{self.kpi_column}' not found in data. --------")
                return {'By Duration': pd.DataFrame()}
                
            # Check if power analysis inputs are valid
            budget_limit = self.power_analysis_inputs.get('Budget')
            cpik = self.power_analysis_inputs.get('Cost')
            
            if not budget_limit or not cpik:
                print("-------- Budget or cost per unit not provided for power analysis. --------")
                return {'By Duration': pd.DataFrame()}
                
            try:
                budget_limit = float(budget_limit)
                cpik = float(cpik)
            except (ValueError, TypeError):
                print("-------- Invalid budget or cost values for power analysis. --------")
                return {'By Duration': pd.DataFrame()}
                
            # Check if we have matched markets in the session state
            if 'matched_markets' not in st.session_state or st.session_state['matched_markets'] is None:
                print("-------- No matched markets available for power analysis. --------")
                return {'By Duration': pd.DataFrame()}
                
            matched_markets = st.session_state['matched_markets']
            if matched_markets.empty:
                print("-------- Empty matched markets for power analysis. --------")
                return {'By Duration': pd.DataFrame()}
            
            # Check if required columns exist in matched markets
            req_cols = ['Test Market Identifier', 'Control Market Identifier']
            if not all(col in matched_markets.columns for col in req_cols):
                print("-------- Required columns missing in matched markets data. --------")
                return {'By Duration': pd.DataFrame()}
            
            # Extract market data safely
            try:
                # Extract the relevant data (Market and KPI column) from the dataframe
                kpi_data = self.kpi_df[['Market', self.kpi_column]]
                
                # Extract the list of similar markets (Test Market and Control Market pairs)
                markets = matched_markets[req_cols]
                l_markets = markets.values.tolist()
                test_market = matched_markets['Test Market Identifier'].values.tolist()
                control_market = matched_markets['Control Market Identifier'].values.tolist()
            except Exception as e:
                print(f"-------- Error extracting market data: {str(e)} --------")
                return {'By Duration': pd.DataFrame()}
            
            # Check if we have lifts in the parameters
            lifts = self.power_analysis_parameters.get('Lifts', [15])
            if not lifts:
                print("-------- No lift percentages specified for power analysis. --------")
                return {'By Duration': pd.DataFrame()}
            
            # Check power and alpha
            power = self.power_analysis_parameters.get('Power', 0.8)
            alpha = self.power_analysis_parameters.get('Alpha', 0.1)
            
            testing_markets, results = [], []
            
            # Iterate over each market pair
            for market in l_markets:
                try:
                    # Append the current market to the testing markets list
                    testing_markets.append(market)
    
                    # Flatten the list of testing markets
                    geo_input = [item for sublist in testing_markets for item in sublist]
    
                    # Filter the KPI data for the selected markets
                    geo_data = kpi_data[kpi_data['Market'].isin(geo_input)]
    
                    if geo_data.empty:
                        print(f"-------- No KPI data for markets {geo_input}. --------")
                        continue
                    
                    # Calculate the mean and standard deviation for the KPI
                    kpi_mean = geo_data[self.kpi_column].mean()
                    kpi_std = geo_data[self.kpi_column].std()
    
                    if kpi_mean == 0 or pd.isna(kpi_mean) or pd.isna(kpi_std):
                        print(f"-------- Invalid KPI statistics for markets {geo_input}. --------")
                        continue
                    
                    # Perform power analysis for each lift percentage
                    for lift in lifts:
                        try:
                            # Convert lift to decimal
                            lift_decimal = float(lift) / 100
                            
                            # Calculate effect size (Cohen's d)
                            effect_size = (kpi_mean * lift_decimal) / kpi_std if kpi_std > 0 else 0
                            
                            if effect_size <= 0:
                                print(f"-------- Invalid effect size for lift {lift}%. --------")
                                continue
                            
                            # Initialize power analysis
                            power_analysis = TTestIndPower()
                            
                            # Calculate sample size
                            sample_size = power_analysis.solve_power(
                                effect_size=effect_size,
                                power=power,
                                alpha=alpha,
                                ratio=1.0,
                                alternative='two-sided'
                            )
                            
                            # Round up to nearest integer
                            sample_size = math.ceil(sample_size)
                            
                            # Calculate test cost
                            cost = sample_size * cpik
                            
                            # Map sample size to time duration based on data granularity
                            if self.date_granularity == 'Daily':
                                duration = f"{sample_size} days"
                            elif self.date_granularity == 'Weekly':
                                duration = f"{sample_size} weeks"
                            elif self.date_granularity == 'Monthly':
                                duration = f"{sample_size} months"
                            else:
                                duration = f"{sample_size} units"
                            
                            # If the cost is within budget, add to results
                            if cost <= budget_limit:
                                results.append({
                                    'Number of Test Markets': len(testing_markets),
                                    'Markets': ', '.join([str(m) for m in test_market[:len(testing_markets)]]),
                                    'Control Markets': ', '.join([str(m) for m in control_market[:len(testing_markets)]]),
                                    'Lift': f"{lift}%",
                                    'Sample Size': sample_size,
                                    'Duration': duration,
                                    'Cost': f"${cost:,.2f}",
                                    'Budget Utilization': f"{(cost/budget_limit)*100:.2f}%"
                                })
                        except Exception as e:
                            print(f"-------- Error in lift analysis {lift}%: {str(e)} --------")
                            continue
                    
                    # Break if we've processed all markets
                    if len(testing_markets) >= len(l_markets):
                        break
                except Exception as e:
                    print(f"-------- Error processing market pair {market}: {str(e)} --------")
                    continue
    
            # Convert results to DataFrame
            results_df = pd.DataFrame(results) if results else pd.DataFrame()
            
            # Return a dictionary with results
            return {'By Duration': results_df}
            
        except Exception as e:
            print(f"-------- Error in power_analysis: {str(e)}. --------")
            return {'By Duration': pd.DataFrame()}
