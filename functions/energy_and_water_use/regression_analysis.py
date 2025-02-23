"""Functions for  regression analysis and prediction of data center specifications."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from statsmodels.formula.api import mixedlm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_col_and_log_transform_histograms(df: pd.DataFrame, column_name: str, color: str, figure_dir: Path) -> None:
    """Plot histograms of a column and its log-transformed values and saves the plot under the figure directory."""
    # Initialize the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of the specified column
    axes[0].hist(df[column_name], bins=50, color=color, edgecolor="black")
    axes[0].set_xlabel(f"{column_name} per data center")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Histogram of {column_name}")

    # Histogram of log-transformed column
    axes[1].hist(np.log1p(df[column_name]), bins=50, color=color, edgecolor="black")
    axes[1].set_xlabel(f"log {column_name} per data center")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of log {column_name}")

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{figure_dir}/{column_name}_histograms.png")
    plt.close(fig)


def predict_white_space_from_total_space(
    data_centers_baseline: pd.DataFrame,
    data_centers_power_scenario: pd.DataFrame,
    total_space_col_name: str = "total_space_m2",
    white_space_col_name: str = "white_space_m2",
    critical_power_col_name: str = "critical_power_mw",
) -> pd.DataFrame:
    """Predict white space from total space through linear regression.

    Args:
        data_centers_baseline (pd.DataFrame): The data centers DataFrame without imputed values.
        data_centers_power_scenario (pd.DataFrame): The data centers DataFrame with imputed critical power values.
        total_space_col_name (str): The column name for total space (default is "total_space_m2").
        white_space_col_name (str): The column name for white space (default is "white_space_m2").
        critical_power_col_name (str): The column name for critical power (default is "critical_power_mw").

    Returns:
        pd.DataFrame: The DataFrame with predicted white space values.
    """
    # Define the training data, which includes rows with known total and white space values
    training_data = data_centers_baseline[[total_space_col_name, white_space_col_name]].dropna().copy()

    # Log transform the total and white space columns
    x_log = np.log1p(training_data[total_space_col_name].to_numpy()).reshape(-1, 1)
    y_log = np.log1p(training_data[white_space_col_name])

    # Fit the linear regression model
    reg_log = LinearRegression().fit(x_log, y_log)
    logging.info(
        "Fit: log %s = %.4f * log %s + %.4f",
        white_space_col_name,
        reg_log.coef_[0],
        total_space_col_name,
        reg_log.intercept_,
    )

    # Find the R2 score
    r2_log = r2_score(y_log, reg_log.predict(x_log))
    logger.info("R2: %.4f", r2_log)

    # Define the test data: data centers where critical power and white space are unknown, and total space is known
    test_data = (
        data_centers_power_scenario[
            data_centers_power_scenario[[critical_power_col_name, white_space_col_name]].isna().all(axis=1)
        ]
        .dropna(subset=[total_space_col_name])
        .copy()
    )

    # Log transform the total space and white space columns
    for col in [total_space_col_name, white_space_col_name]:
        test_data[f"log_{col}"] = np.log1p(test_data[col])

    # Predict the white space values
    test_data[f"log_{white_space_col_name}"] = reg_log.predict(test_data[[f"log_{total_space_col_name}"]])

    # Convert the log-transformed white space values back to the original scale
    test_data[white_space_col_name] = np.expm1(test_data[f"log_{white_space_col_name}"])

    # Add a boolean column to indicate that the data is predicted
    test_data[f"{white_space_col_name}_predicted"] = True
    data_centers_power_scenario[f"{white_space_col_name}_predicted"] = False

    # Update the original DataFrame with the predicted values
    data_centers_power_scenario.update(test_data[[white_space_col_name, f"{white_space_col_name}_predicted"]])

    # Set the predicted white space values to False for remaining rows
    data_centers_power_scenario[f"{white_space_col_name}_predicted"] = data_centers_power_scenario[
        f"{white_space_col_name}_predicted"
    ].fillna(value=False)

    return data_centers_power_scenario


def plot_regression_diagnostics(y_train: pd.Series, y_pred_train: np.ndarray, save_path: str | None = None) -> None:
    """Plot regression diagnostic plots (actual vs predicted and residuals)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Actual vs Predicted
    ax1.scatter(y_train, y_pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
    ax1.set_title("Actual vs Predicted Values")
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")

    # Residuals
    residuals = y_train - y_pred_train
    ax2.scatter(y_pred_train, residuals, alpha=0.5)
    ax2.axhline(y=0, color="r", linestyle="--")
    ax2.set_title("Residuals Plot")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def polynomial_regression_analysis(
    df: pd.DataFrame,
    target_column: str,
    input_columns: list[str],
    *,
    polynomial_degree: int = 2,
    categorical_columns: list[str] | None = None,
    z_score_threshold: float = 3,
    k_folds: list[int] = (3, 5),
    plot_results: bool = False,
    save_plots: bool = False,
    plot_dir: str | None = None,
) -> pd.DataFrame:
    """Analyze, predict, and validate missing values using polynomial regression.

    Args:
        df: Input DataFrame
        target_column: Column to predict
        input_columns: Features for prediction
        categorical_columns: Optional categorical features
        z_score_threshold: Threshold for outlier removal
        polynomial_degree: Degree for polynomial features
        k_folds: Tuple of number of folds for k-fold cross validation (default: (3, 5))
        plot_results: Whether to generate plots
        save_plots: Whether to save plots
        plot_dir: Directory for saved plots
    """
    # Log transform numeric columns
    numeric_columns = [target_column, *input_columns]
    df_transformed = df.copy()
    df_transformed[numeric_columns] = df[numeric_columns].apply(np.log1p)

    # Split data and handle outliers
    required_cols = [target_column, *input_columns, *(categorical_columns or [])]
    df_train = df_transformed.dropna(subset=required_cols)
    df_predict = df_transformed[df_transformed[target_column].isna()]

    # Calculate and filter outliers
    ratios = pd.DataFrame({f"ratio_{col}": df_train[target_column] / df_train[col] for col in input_columns})
    z_scores = ratios.apply(zscore)
    df_train_clean = df_train[(z_scores.abs() <= z_score_threshold).all(axis=1)]

    # Handle categorical features
    if categorical_columns:
        encoder = OneHotEncoder(drop="first", sparse_output=False)
        encoded_train = encoder.fit_transform(df_train_clean[categorical_columns])
        encoded_predict = encoder.transform(df_predict[categorical_columns])

        cat_features = encoder.get_feature_names_out(categorical_columns)
        encoded_train_df = pd.DataFrame(encoded_train, columns=cat_features)
        encoded_predict_df = pd.DataFrame(encoded_predict, columns=cat_features)

    # Generate polynomial features
    poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    numeric_train = poly.fit_transform(df_train_clean[input_columns])
    numeric_predict = poly.transform(df_predict[input_columns])

    feature_names = poly.get_feature_names_out(input_columns)
    x_train = pd.DataFrame(numeric_train, columns=feature_names)
    x_predict = pd.DataFrame(numeric_predict, columns=feature_names)

    # Combine numeric and categorical features
    if categorical_columns:
        x_train = pd.concat([x_train, encoded_train_df], axis=1)
        x_predict = pd.concat([x_predict, encoded_predict_df], axis=1)

    y_train = df_train_clean[target_column]

    # Train model and predict
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)

    # K-fold cross validation
    if k_folds:
        for k in k_folds:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            mae_scores = []

            for train_idx, test_idx in kf.split(x_train):
                x_fold_train, x_fold_test = x_train.iloc[train_idx], x_train.iloc[test_idx]
                y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

                fold_model = LinearRegression()
                fold_model.fit(x_fold_train, y_fold_train)
                y_fold_pred = fold_model.predict(x_fold_test)
                mae_scores.append(mean_absolute_error(y_fold_test, y_fold_pred))

            logger.info("%d-Fold CV - Mean MAE: %.4f", k, np.mean(mae_scores))

    # Generate plots if requested
    if plot_results:
        save_path = f"{plot_dir}/prediction_analysis.png" if save_plots and plot_dir else None
        plot_regression_diagnostics(y_train, y_pred_train, save_path)

    # Update DataFrame with predictions
    df.loc[df_predict.index, target_column] = np.expm1(model.predict(x_predict))
    df[f"{target_column}_predicted"] = df.index.isin(df_predict.index)

    # Log model performance
    r2 = r2_score(y_train, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    logger.info("R² Score: %.4f", r2)
    logger.info("RMSE: %.4f", rmse)

    return df


def mixed_effects_model_analysis(
    data_centers_df: pd.DataFrame,
    input_cols: str,
    output_cols: str,
    categorical_col: str,
    *,
    display_results: bool = False,
) -> pd.DataFrame:
    """Fit a mixed-effects model to predict the output column from the input column and a categorical variable.

    Args:
        data_centers_df (pd.DataFrame): DataFrame with data center information.
        input_cols (str): Column name for the input variable.
        output_cols (str): Column name for the output variable.
        categorical_col (str): Column name for the categorical variable.
        display_results (bool): Whether to display plots and summary.

    Returns:
        pd.DataFrame: Updated DataFrame with predicted values.
    """
    # Log-transform the input and output columns
    data_centers_df[f"log_{input_cols}"] = np.log1p(data_centers_df[input_cols])
    data_centers_df[f"log_{output_cols}"] = np.log1p(data_centers_df[output_cols])

    # Drop rows with missing values in the columns used in the model
    data_centers_clean = data_centers_df.dropna(subset=[f"log_{input_cols}", f"log_{output_cols}", categorical_col])

    # Mixed-effects model with random intercepts for the categorical variable
    model = mixedlm(
        f'Q("log_{output_cols}") ~ Q("log_{input_cols}")',
        data_centers_clean,
        groups=data_centers_clean[categorical_col],
        re_formula="~1",
    )

    # Fit the model
    result = model.fit()

    # Get the predicted values (fitted values)
    predicted_values = result.fittedvalues

    # Get the observed values (true values)
    observed_values = data_centers_clean[f"log_{output_cols}"]

    # Calculate residuals (observed - predicted)
    residuals = observed_values - predicted_values

    # Log model summary
    logger.info(result.summary())

    # Log r2 and mse
    r2_mixed = r2_score(observed_values, predicted_values)
    mse_mixed = mean_squared_error(observed_values, predicted_values)
    logger.info("Mixed-Effects Model R2: %f", r2_mixed)
    logger.info("Mixed-Effects Model MSE: %f", mse_mixed)

    if display_results:
        # Log model summary
        logger.info(result.summary())

        # Log r2 and mse
        r2_mixed = r2_score(observed_values, predicted_values)
        mse_mixed = mean_squared_error(observed_values, predicted_values)
        logger.info("Mixed-Effects Model R2: %f", r2_mixed)
        logger.info("Mixed-Effects Model MSE: %f", mse_mixed)

        # Plotting
        plt.figure(figsize=(14, 6))

        # Predicted vs Observed
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=(observed_values), y=(predicted_values))
        plt.plot(
            [(observed_values).min(), (observed_values).max()],
            [(observed_values).min(), (observed_values).max()],
            "--",
            color="red",
        )  # Diagonal line (perfect prediction)
        plt.xlabel("Observed log Values")
        plt.ylabel("Predicted log Values")
        plt.title("log Predicted vs logObserved")

        # Residuals Plot
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=predicted_values, y=residuals)
        plt.axhline(0, color="red", linestyle="--")  # Zero line for residuals
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")

        # Histogram of residuals
        plt.figure(figsize=(7, 5))
        sns.histplot(residuals, kde=True, color="skyblue", bins=30)
        plt.axvline(0, color="red", linestyle="--")  # Zero line for residuals
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residuals")

        # Show all plots
        plt.tight_layout()
        plt.show()

    # Filter the rows where critical power is missing
    missing_data = data_centers_df[data_centers_df[f"log_{output_cols}"].isna()]

    # Use the trained model to make predictions
    predictions = result.predict(missing_data)

    # Add the predicted values to the dataframe
    missing_data[f"Mixed-effect log_{output_cols}"] = predictions

    # Transform back to the original scale (exponentiate the log predictions)
    missing_data[f"Mixed-effect {output_cols}"] = np.expm1(missing_data[f"Mixed-effect log_{output_cols}"])

    # Update the output column with the predicted values
    data_centers_df.loc[missing_data.index, output_cols] = missing_data[f"Mixed-effect {output_cols}"]

    # Add a column to indicate which values were predicted
    data_centers_df[f"Mixed-effect {output_cols} Predicted"] = False
    data_centers_df.loc[missing_data.index, f"Mixed-effect {output_cols} Predicted"] = True

    # Drop the log-transformed columns
    data_centers_df = data_centers_df.drop(columns=[f"log_{input_cols}", f"log_{output_cols}"])

    return data_centers_df
