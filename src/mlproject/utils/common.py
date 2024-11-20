import os
from box.exceptions import BoxValueError
import yaml
from mlproject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads yaml file and reurns.

    Args:
        path_to_yaml (Path): path-like input.

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f".yaml file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(".yaml file is empty.")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool=True):
    """Create list of directories.

    Args:
        path_to_directories (list): list of path of directory.
        verbose (bool, optional): Ignore if multiple directories is to be created. Defaults to True.
    """
    try:
        for dirs in path_to_directories:
            os.makedirs(dirs, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at {dirs}")
    except Exception as e:
        raise f"Error creating directories: {e}"
    
@ensure_annotations
def save_json(path: Path, data: dict):
    """Save json data.

    Args:
        path (Path): path to json file.
        data (dict): Data to be saved in json file.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")
    except Exception as e:
        raise f"Error saving json file: {e}"
    
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load json file.

    Args:
        path (Path): Path to json file.

    Returns:
        ConfigBox: Data as class aributes instead of dict
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise f"Error loading json file: {e}"
    
@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary file.

    Args:
        data (Any): Data to be saved as binary.
        path (Path): Path to binary file.
    """
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at {path}")
    except Exception as e:
        raise f"Error saving bin: {e}"
    
@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary file.

    Args:
        path (Path): Path to binary file.

    Returns:
        Any: object stored in the file        
    """
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded from {path}")
        return data
    except Exception as e:
        raise f"Error loading binary file: {e}"
    
@ensure_annotations
def get_size(path: Path) -> str:
    """Get size in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size in KB.
    """
    try:
        size_in_kb = round(os.path.getsize(path)/1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        raise f"Error getting size: {e}"
    

@ensure_annotations
def get_missing_columns(df: pd.DataFrame):
    """
    This function identifies columns in a DataFrame and
    calculates the percentage of missing values for those columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: A list of numeric and categorical columns with missing values.
        dict: A dictionary containing the percentage of missing values
              for each column with missing data.
    """
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    category_columns = df.select_dtypes(include=['object']).columns
    
    numeric_columns_with_na = [col for col in numeric_columns if df[col].isna().sum() > 0]   
    category_columns_with_na = [col for col in category_columns if df[col].isna().sum() > 0]   
    
    # Dictionary to store percentage of missing values for each column
    numeric_columns_missing_data_info = {
        column: f"{np.round(df[column].isnull().mean() * 100, 3)}% missing values"
        for column in numeric_columns_with_na
    }
    
    category_columns_missing_data_info = {
        column: f"{np.round(df[column].isnull().mean() * 100, 3)}% missing values"
        for column in category_columns_with_na
    }
    
    return numeric_columns_with_na, category_columns_with_na, numeric_columns_missing_data_info, category_columns_missing_data_info
    
@ensure_annotations
def get_outliers(df, column, lower_quantile=0.25, upper_quantile=0.75, iqr_multiplier=1.5) -> pd.Series:
    """
    Identify outliers in a DataFrame column using the IQR method with customizable quantiles.

    Parameters:
    - df: pandas DataFrame
    - column: column name as a string
    - lower_quantile: float, the lower quantile value (default is 0.25 for the 25th percentile)
    - upper_quantile: float, the upper quantile value (default is 0.75 for the 75th percentile)
    - iqr_multiplier: float, multiplier for the IQR to define outlier bounds (default is 1.5)

    Returns:
    - A boolean Series where True indicates an outlier.
    """

    # Calculate the specified quantiles
    percentile_lower = np.quantile(df[column], lower_quantile)
    percentile_upper = np.quantile(df[column], upper_quantile)

    # Calculate the Interquartile Range (IQR)
    IQR = percentile_upper - percentile_lower

    # Calculate the lower and upper bounds for outliers
    lower_bound = percentile_lower - (iqr_multiplier * IQR)
    upper_bound = percentile_upper + (iqr_multiplier * IQR)

    # Return the boolean Series indicating outliers
    return (df[column] < lower_bound) | (df[column] > upper_bound)

    
@ensure_annotations
def cross_tab_df(df: pd.DataFrame, cat_col: str, target_cat_col: str) -> pd.DataFrame:
    """Creates a crosstab of the specified categorical column against the target column,
    and calculates the percentage for one of the target column values.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_col (str): Column name of the categorical feature to group by (e.g., 'Security_Type').
        target_cat_col (str): Categorical Column name of the target variable to calculate distribution (e.g., 'Status_str'). Must have two unique values.

    Raises:
        ValueError: If there’s an error in creating the crosstab.

    Returns:
        pd.DataFrame: Crosstab with counts and percentage of a specific target value.
    
    """
    try:
        # Create the crosstab (counts for each combination of cat_col and target_col values)        
        cross_tab = pd.crosstab(df[cat_col], df[target_cat_col])
        
        # Ensure unique target column values for dynamic naming
        target_values = df[target_cat_col].unique()
        
        if len(target_values) != 2:
            raise ValueError("The target column must have exactly two unique values.")

        # Calculate the percentage of the first target column value (e.g., 'Default')
        main_target = target_values[0]
        other_target = target_values[1]

        cross_tab[f'Percentage_{main_target}'] = (
            (cross_tab[main_target] / (cross_tab[main_target] + cross_tab[other_target])) * 100
        ).round(2).astype(str) + '%'
        
        return cross_tab
    except Exception as e:
        raise ValueError('Error creating cross tab.') from e
    
@ensure_annotations
def grouped_cross_tab_bar(cross_tab_melted: pd.DataFrame, x_column: str, y_column: str, group_by: str):
    """_summary_

    Args:
        cross_tab_melted (pd.DataFrame): _description_
        x_column (str): _description_
        y_column (str): _description_
        group_by (str): _description_

    Raises:
        ValueError: _description_
    """
    try:
        # Step 2: Create the grouped bar chart with Plotly
        fig = px.bar(
            cross_tab_melted,
            x=x_column,
            y=y_column,
            color=group_by,
            barmode='group',
            title=f'Distribution of {x_column} with {y_column}',
            labels={x_column: x_column, y_column: y_column},
            # color_discrete_map={'Default': '#87CEEB', 'Non default': '#FFA07A'}  # Colors for each category
        )

        # Customize layout
        fig.update_layout(
            title_font=dict(size=20, color='#005ce6'),
            xaxis=dict(title=x_column),
            yaxis=dict(title=y_column),
            bargap=0.2  # Space between bars in each group
        )

        # Display the plot
        fig.show()
        
    except Exception as e:
        raise ValueError('Error ploting group cross tab bar.') from e

@ensure_annotations
def plot_histogram(
    df: pd.DataFrame, 
    column_name: str, kde=False,
    color = px.colors.sequential.Sunset_r
    ):
    """
    Plots a histogram for a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): Data containing the values to plot.
        column_name (str): Column name for which to plot the histogram.
        color (str or list): Color or color scale for the histogram bars.
        title (str): Title of the plot.
    """
    try:
        
        if kde:
            sns.histplot(df, x=column_name, kde=True)
            plt.show()
        else:
            # Plot histogram using Plotly Express
            fig = px.histogram(
                df, 
                x=column_name, 
                color_discrete_sequence=color if isinstance(color, list) else [color],
                title=f"{column_name} Histogram"
            )
                    
            # Update layout
            fig.update_layout(
                title=dict(text=f"{column_name} Histogram", font=dict(size=20), x=0.5),
                xaxis_title=column_name,
                yaxis_title="Frequency",
                template='plotly_white',
                width=800,
                height=500
            )
            fig.show()
        
    except Exception as e:
        raise ValueError("Error in plotting histogram.") from e
    
@ensure_annotations
def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str, group_by: str):
    """Plot scatter plot by group_by column.

    Args:
        df (pd.DataFrame): _description_
        x_column (str): _description_
        y_column (str): _description_
        group_by (str): _description_

    Raises:
        ValueError: _description_
    """
    try:
        fig = px.scatter(
            data_frame=df, 
            x=df[x_column], 
            y=df[y_column], 
            color=group_by,
            opacity=0.5,
            title=f"{x_column} Vs. {y_column} by {group_by}",
            labels={x_column: x_column, y_column: y_column}
        )

        fig.update_layout(
            # title=title,
            xaxis_title=x_column,
            yaxis_title=y_column,
            title_font=dict(size=20, color='#005ce6'),
        )

        fig.show()
    except Exception as e:
        raise ValueError("Error in plotting scatter plot.") from e
      
@ensure_annotations
def plot_grouped_bar_chart(data: pd.DataFrame, categorical_columns, hue_column: str, num_cols: int=2, ylabel: str='Count', palette: str="pastel", log_scale: bool=False, bar_width: float=0.8):
    """
    Plots a grouped (side-by-side) bar chart for a categorical column grouped by another categorical column.

    Parameters:
    data (DataFrame): The DataFrame containing the data.
    x_column (str): The name of the primary categorical column to plot on the x-axis.
    hue_column (str): The name of the categorical column to group by for side-by-side bars (default is 'Status_str').
    xlabel (str): The label for the x-axis.
    title (str): The title of the plot.
    ylabel (str): The label for the y-axis (default is "Count").
    figsize (tuple): The size of the figure (default is (8, 6)).
    palette (str): The color palette for the plot (default is "pastel").
    log_scale (bool): Whether to use a logarithmic scale for the y-axis (default is False).
    bar_width (float): The width of the bars (default is 0.8, where 1.0 is full width with no spacing).

    Returns:
    None: Displays a grouped bar chart.
    """
    try:
        rows = len(categorical_columns) // num_cols + (len(categorical_columns) % num_cols > 0)
        
        plt.figure(figsize=(10, 4 * rows))
        plt.suptitle("Categorical Features Vs Categorical Column", fontsize=20, fontweight='bold', y=0.85)
        
        # Setting the style and palette
        sns.set(style="whitegrid")
        
        for i, col in enumerate(categorical_columns):
            plt.subplot(rows, num_cols, i + 1)
            # Creating the grouped bar plot
            sns.histplot(data=data, x=col, hue=hue_column, multiple="dodge", palette=palette, shrink=bar_width)
            
            # Adding a title and labels
            plt.title(f'{col} Vs {hue_column}', fontsize=16)
            plt.xlabel(col, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
        
            if log_scale:
                plt.yscale('log')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    except Exception as e:
        raise e
       
@ensure_annotations
def plot_bar(df: pd.DataFrame, x_column: str, y_column: str ):
    """Plots a bar chart using Plotly, with values labeled on top of each bar.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_column (str): Name of the column for the x-axis.
        y_column (str): Name of the column for the y-axis.

    Raises:
        ValueError: If there is an error in plotting the bar graph.
    """
    try:
        fig = px.bar(
            data_frame=df,
            x=x_column,
            y=y_column,
            title=f'{y_column} by {x_column}',
            labels={y_column: y_column},
            color_discrete_sequence=['indianred']
        )

        # Show the percentage value on top of each bar
        fig.update_traces(text=df[y_column].round(2), textposition='outside')
        fig.show()
    except Exception as e:
        raise ValueError("Error plotting bar graph:", e) from e
     
@ensure_annotations
def plot_univariate_distribution_categorical_features(df: pd.DataFrame, categorical_features):
    """Categorical features distribution. Count plot for each category.

    Args:
        df (pd.DataFrame): The dataframe having the categorical columns.
        categorical_features: Categorical features column names.

    Raises:
        f: _description_
    """
    try:
        # Calculate the number of rows needed
        rows = len(categorical_features) // 2 + len(categorical_features) % 2

        # Create subplots
        fig = make_subplots(rows=rows, cols=2, subplot_titles=[f'{col} Distribution' for col in categorical_features],
                            vertical_spacing=0.03)

        # Loop through categorical features to add bar plots
        for i, col in enumerate(categorical_features):
            row = i // 2 + 1
            col_idx = i % 2 + 1
            # Count the occurrences of each category
            category_count = df[col].value_counts()
            fig.add_trace(go.Bar(x=category_count.index, y=category_count.values, name=col, text=category_count.values, textposition='auto'), row=row, col=col_idx)

        # Adjust layout
        fig.update_layout(height=300 * rows, width=1000, title_text="Categorical Feature Distributions", showlegend=False)

        # Show the plot
        fig.show()
    except Exception as e:
        raise ValueError("Error in plotting univariate distribution of dataframe.") from e

@ensure_annotations
def plot_univariate_distribution_num_features(df: pd.DataFrame, numerical_features):
    """Numerical features distribution. Using plotly violin.

    Args:
        df (pd.DataFrame): _description_
        numerical_features (list): _description_

    Raises:
        f: _description_
    """
    try:
        # Calculate the number of rows needed
        rows = len(numerical_features)

        # Create subplots
        fig = make_subplots(rows=rows, cols=1, subplot_titles=[f'{col} Distribution' for col in numerical_features],
                            vertical_spacing=0.03, horizontal_spacing=0.1)

        # Loop through numerical features to add horizontal box plots
        for i, col in enumerate(numerical_features):
            row = i + 1
            fig.add_trace(
                go.Box(x=df[col], name=col, boxmean=True, orientation='h'),  # 'orientation=h' for horizontal
                row=row,
                col=1
            )
            
        # Adjust layout
        fig.update_layout(height=500 * rows, width=1000, title_text="Numerical Feature Distributions", showlegend=False)
        fig.update_layout(template="plotly", grid=dict(rows=rows, columns=2))

        # Show the plot
        fig.show()

    except Exception as e:
        raise e

@ensure_annotations
def plot_numerical_histograms(df, numerical_features, bins=20):
    """
    Plots histograms for numerical features to show data distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numerical_features (list): List of numerical feature column names to plot.
        bins (int): Number of bins for the histogram.
    """
    try:
        # Calculate the number of rows needed
        rows = len(numerical_features)

        # Create subplots with one column
        fig = make_subplots(
            rows=rows, 
            cols=1, 
            subplot_titles=[f'{col} Histogram' for col in numerical_features],
            vertical_spacing=0.03  # Adjust vertical spacing
        )

        # Loop through numerical features to add histograms
        for i, col in enumerate(numerical_features):
            row = i + 1

            # Add histogram plot
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    nbinsx=bins,
                    name=col,
                    marker=dict(color='skyblue', line=dict(width=1, color='black'))  # Optional styling
                ),
                row=row,
                col=1
            )

        # Adjust layout
        fig.update_layout(
            height=300 * rows,  # Adjust height dynamically based on the number of rows
            width=1000,  # Set fixed width
            title_text="Numerical Feature Histograms",
            showlegend=False,
            template="plotly",
            margin=dict(t=50, l=30, r=30, b=30)  # Optional: Adjust margins
        )

        # Show the plot
        fig.show()

    except Exception as e:
        raise e

@ensure_annotations
def univariate_numeric_imputation(df: pd.DataFrame, numeric_columns, method: str = 'mean'):
    """
    Impute missing values in numerical columns with specified method
    and visualize the distribution before and after imputation.

    Args:
        df (pd.DataFrame): DataFrame containing numerical columns with missing values.
        numeric_columns (list): List of numerical columns to consider.
        method (str): Imputation method ('mean', 'median', or 'interpolate').
    """
    try:
        # Filter columns with missing values
        num_cols_with_na = [col for col in numeric_columns if df[col].isna().sum() > 0]

        # If no columns with missing values, exit
        if not num_cols_with_na:
            return "No numerical columns with missing values in DataFrame."

        # Create subplots to display before and after distributions
        rows = len(num_cols_with_na)
        fig, axes = plt.subplots(rows, 1, figsize=(10, rows * 4))
        fig.suptitle("Distributions Before and After Imputation", fontsize=20, fontweight='bold', y=1.02)

        # Handle single-column case (axes becomes a single plot object, not an array)
        if rows == 1:
            axes = [axes]

        for i, col in enumerate(num_cols_with_na):
            ax = axes[i]
            
            # Plot original distribution
            df[col].plot(kind="kde", ax=ax, label="Original", color='blue')
            
            if f"{col}_{method}_imputed" in df.columns:
                return f"{col}_{method}_imputed already exists in DataFrame." 

            # Imputation logic
            if method == 'mean':
                imputed_value = df[col].mean()
                df[f"{col}_{method}_imputed"] = df[col].fillna(imputed_value)
            elif method == 'median':
                imputed_value = df[col].median()
                df[f"{col}_{method}_imputed"] = df[col].fillna(imputed_value)
            elif method == 'interpolate':
                # Direct interpolation
                df[f"{col}_{method}_imputed"] = df[col].interpolate(method='linear')
            else:
                raise ValueError(f"Invalid method '{method}'. Choose 'mean', 'median', or 'interpolate'.")

            # Plot after-imputation distribution
            df[f"{col}_{method}_imputed"].plot(kind="kde", ax=ax, label=f"After Imputation ({method})", color='orange')

            # Add titles, labels, and legends
            ax.set_title(f"{col} - Before and After Imputation ({method})", fontsize=14)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

        # Final layout adjustments
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

@ensure_annotations
def plot_bivariate_distribution_num_vs_cat(df: pd.DataFrame, numeric_features, cat_column_name: str, num_col: int=2):
    """_summary_

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        numeric_features (Index): Categorical columns in dataframe.
        cat_column_name (str): Categorical target column to plot againt.
        num_col (int, optional): Number of plot in a row. Defaults to 2.
        num_row (int, optional): Number of rows. Defaults to 2.

    Raises:
        ValueError: _description_
    """
    try:
        # Calculate the number of rows required
        rows = len(numeric_features) // num_col + (len(numeric_features) % num_col > 0)
        
        # Dynamically adjust the figure height based on the number of rows
        plt.figure(figsize=(8, 5 * rows))  # Adjust 5 to control height per row
        plt.suptitle("Numerical Features Vs Categorical Column", fontsize=20, fontweight='bold', y=1.02)


        for i, col in enumerate(numeric_features):
            plt.subplot(rows, num_col, i + 1)
            sns.boxplot(x=cat_column_name, y=col, data=df, medianprops=dict(color='red'), linewidth=2.5)
            plt.title(f'{col} vs {cat_column_name}', fontsize=16)
            plt.xlabel(cat_column_name, fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the top spacing to make room for suptitle
        plt.show()
    except Exception as e:
        raise e
    
@ensure_annotations
def before_after_distribution(df: pd.DataFrame, before_column: str, after_column: str):
    """
    Compare the distribution of a column before and after imputation using KDE plots.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        before_column (str): Name of the column before imputation.
        after_column (str): Name of the column after imputation.
    """
    try:
        plt.figure(figsize=(10, 6))
        # Plot original distribution
        df[before_column].plot(kind="kde", label="Original", color='blue', linewidth=2)
        # Plot after-imputation distribution
        df[after_column].plot(kind="kde", label="After imputation", color='orange', linewidth=2)
        
        plt.title(f"Distribution of {before_column} vs {after_column}", fontsize=14)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()    
    except Exception as e:
        raise e
    
@ensure_annotations
def plot_corr_matrix_num_features(df: pd.DataFrame, numerical_features, method: str='pearson'):
    """Correlation matrix of numerical features.

    Args:
        df (pd.DataFrame): _description_
        numerical_features (list): _description_
        method (str, optional): ['pearson', 'kendall', 'spearman']. Defaults to 'pearson' Good for linear relationship. 

    Raises:
        ValueError: _description_
    """
    try:
        pearson_corr_metrix_num = df[numerical_features].corr(method)

        fig = go.Figure(data=go.Heatmap(
            z=pearson_corr_metrix_num.values,  # Correlation values
            x=pearson_corr_metrix_num.columns,  # X-axis labels (features)
            y=pearson_corr_metrix_num.index,  # Y-axis labels (features)
            colorscale='Blues', # color scale for heatmap
            zmin=-1,  # Set the color scale range
            zmax=1,   # Set the color scale range
            colorbar=dict(title='Correlation'),  # Add a color bar with a title
            text=np.round(pearson_corr_metrix_num.values, 2),  # Display values rounded to 2 decimal places
            texttemplate="%{text}",  # Use the text for display in each box
        ))

        # Add layout and titles
        fig.update_layout(
            title='Pearson Correlation Matrix of Numerical Features',
            xaxis_nticks=36,  # Adjust the number of ticks on the x-axis if needed
            height=600,
            width=800
        )

        # Show the plot
        fig.show()
    except Exception as e:
        raise ValueError("Error in plotting pearson correlation of dataframe.") from e
    
    

# @ensure_annotations
# def plot_count_distribution(df: pd.DataFrame, column_name: str):
#     """Plots the count distribution of a specified categorical column in a DataFrame

#     Args:
#         df (pd.DataFrame): The DataFrame containing the data.
#         column_name (str): The name of the categorical column to plot.
#         color (str, optional): _description_. Defaults to 'purple'.

#     Raises:
#         ValueError: _description_
#         f: _description_
        
#     Returns:
#         None: Displays a count plot.
#     """
#     try:
#         # Confirm column is a caegorical column
#         if df[column_name].dtype != "object":
#             raise ValueError("Column must be a categorical colum.")
        
#         # First, count the occurrences of each unique value in the column
#         counts = df[column_name].value_counts().reset_index()
#         counts.columns = [f'{column_name} Status', 'Count']  # Rename the columns for clarity

#         # Create the bar chart
#         fig = px.bar(counts, x=f'{column_name} Status', y='Count', title=f'{column_name} Count', color=f'{column_name} Status',
#                     labels={f'{column_name} Status': f'{column_name} Status'}, text_auto=True,
#                     color_discrete_sequence=px.colors.sequential.Sunset_r
#                     )

#         # Customize the plot
#         fig.update_layout(
#             title_font=dict(size=20, color='orange'), 
#             xaxis_title=f'{column_name} Status',
#             yaxis_title="Count",
#             # template='plotly_dark',  # Dark template
#             width=800,  # Width of the plot
#             height=500,  # Height of the plot
#             bargap=0.2,  # Adjust bar gap
#         )

#         # Show the plot
#         fig.show()
#     except Exception as e:
#         raise ValueError("Error in plotting univariate count distribution of dataframe.") from e
    
# @ensure_annotations
# def plot_bivariate_distribution_categorical_features(df: pd.DataFrame, categorical_features, column_name: str, cols: int=2):
#     """Categorical features Vs column_name.

#     Args:
#         df (pd.DataFrame): _description_
#         categorical_features: Categorical features.
#         column_name (str): Column to plot distribution against.

#     Raises:
#         f: _description_
#     """
#     try:
#         # Number of categorical features
#         num_cats = len(categorical_features)

#         # Calculate number of rows and columns for subplots
#         rows = math.ceil(num_cats / cols)  # Dynamically calculate rows needed

#         # Create figure with space between plots
#         fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
#         fig.subplots_adjust(hspace=0.5)  # Add space between rows

#         for i, col in enumerate(categorical_features):
#             ax = axes.flat[i]  # Access the subplot axis

#             # Filter out rows where the column has missing or empty values
#             df_filtered = df[df[col].notna()]

#             # Check if there are any valid values left after filtering
#             if df_filtered.empty:
#                 ax.set_title(f"No valid data for {col}")
#                 continue

#             # Calculate column counts for each category
#             try:
#                 counts = df_filtered.groupby(col)[column_name].value_counts(normalize=True).unstack()
#             except ValueError as e:
#                 print(f'Error while calculating {column_name} count:', e)

#             # Calculate column percentage
#             counts[f'{column_name}_percentage'] = counts[column_name] * 100

#             # Resetting index for plotting
#             counts = counts.reset_index()

#             # Plotting
#             try:
#                 sns.barplot(x=col, y=f'{column_name}_percentage', data=counts, color='#FC8D62', ax=ax)

#                 ax.set_title(f'{column_name} Percentage by {col}')
#                 ax.set_xlabel(col)
#                 ax.set_ylabel(f'{column_name} Percentage')
#                 ax.set_xticks(ax.get_xticks())  # Ensure correct x-axis ticks
#                 ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate for better readability

#                 # Adding percentages below the bars
#                 for p in ax.patches:
#                     percentage = f'{p.get_height():.2f}'  # Format percentage to 2 decimal places
#                     ax.annotate(percentage,
#                                 (p.get_x() + p.get_width() / 2., p.get_height()),  # Positioning above the bar
#                                 ha='center', va='bottom', fontsize=11, color='black',
#                                 xytext=(0, -10), textcoords='offset points')  # Adjusting vertical position
#             except ValueError as e:
#                 print("Error ploting:", e)

#         # If there are fewer plots than subplots, hide empty subplots
#         for j in range(i + 1, len(axes.flat)):
#             axes.flat[j].set_visible(False)

#         plt.tight_layout()
#         plt.show()
#     except Exception as e:
#         raise ValueError("Error in plotting bivariate distribution of categorical data.") from e
  

# @ensure_annotations
# def plot_bivariate_distribution_numerical_features(df: pd.DataFrame, numerical_features, column_name: str, rows: int, cols: int, palette = sns.color_palette('Set2')):
#     """Numerical features Vs column_name. Using seaborn boxplot.

#     Args:
#         df (pd.DataFrame): _description_
#         numerical_features: Numerical features.
#         column_name (str): Column to plot distribution against.

#     Raises:
#         f: _description_
#     """
#     try:
#         plt.figure(figsize=(18, 25))
#         for i, col in enumerate(numerical_features):
#             plt.subplot(rows, cols, index=i + 1)
#             sns.boxplot(x=column_name, y=col, data=df, palette=palette)
#             plt.title(f'{col} vs {column_name}')
#             plt.grid(True)

#         plt.tight_layout()
#         plt.show()
#     except Exception as e:
#         raise ValueError("Error in plotting univariate distribution of numerical data.") from e
     

# @ensure_annotations
# def plot_univariate_distribution_num_features(df: pd.DataFrame, numerical_features):
#     """Numerical features distribution. Using plotly violin.

#     Args:
#         df (pd.DataFrame): _description_
#         numerical_features (list): _description_

#     Raises:
#         f: _description_
#     """
#     try:
#         # Calculate the number of rows needed
#         rows = len(numerical_features) // 2 + len(numerical_features) % 2

#         # Create subplots
#         fig = make_subplots(rows=rows, cols=2, subplot_titles=[f'{col} Distribution' for col in numerical_features])

#         # Loop through numerical features to add violin plots
#         for i, col in enumerate(numerical_features):
#             row = i // 2 + 1
#             col_idx = i % 2 + 1
#             fig.add_trace(go.Violin(y=df[col], name=col, box_visible=True, meanline_visible=True), row=row, col=col_idx)

#         # Adjust layout
#         fig.update_layout(height=1000, width=1000, title_text="Numerical Feature Distributions", showlegend=False)
#         fig.update_layout(template="plotly", grid=dict(rows=rows, columns=2))

#         # Show the plot
#         fig.show()

#     except Exception as e:
#         raise ValueError("Error in plotting univariate distribution of dataframe.") from e
  