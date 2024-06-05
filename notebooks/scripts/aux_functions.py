import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def generate_columns(stock: str, contains_vol: bool, contains_USD: bool):
    """
    Generates a list of column names based on the provided stock symbol,
    whether volatility data is included, and whether USD data is included.

    Returns:
        list: A list of column names.
    """
    columns = []
    columns.append(f"{stock}_log_rets")

    if contains_vol:
        columns.append(f"{stock}_gk_vol")

    if contains_USD:
        columns.append(f"USD_log_rets")
        columns.append(f"USD_gk_vol")

    return columns


def save_as_pickle(
    data,
    resultsroute: str,
    model_type: str,
    tablename: str,
    criterion: str,
    type_save: str,
):
    """
    Saves data as a pickle file with a descriptive filename based on the provided parameters.
    """
    with open(
        os.path.join(
            resultsroute,
            f"""{model_type}_{tablename}_{criterion}_best_{type_save}.pickle""",
        ),
        "wb",
    ) as output_file:
        pickle.dump(data, output_file)


def get_all_results_matching(resultsroute: str, substrings: list):
    """
    Finds all pickle files in a directory that contain all of the provided substrings in their filenames.

    Returns:
        dict: A dictionary mapping filenames to their corresponding file paths,
            or an empty dictionary if no matching files are found.
    """
    all_results = {}

    for filename in os.listdir(resultsroute):
        file_path = os.path.join(resultsroute, filename)
        coincidences = 0
        if os.path.isfile(file_path):
            for substring in substrings:
                if substring in filename:
                    coincidences += 1
            if coincidences == len(substrings):
                all_results[filename] = file_path

    print(all_results)
    return all_results


def subset_of_columns(df: pd.DataFrame, substring: str, exclude: str = None):
    """
    Filters a DataFrame to include only columns containing a given substring,
    optionally excluding columns containing another substring.

    Returns:
    pd.DataFrame: A new DataFrame containing only the filtered columns.
    """
    filtered_columns = [col for col in df.columns if substring in col]

    if exclude is not None:
        filtered_columns = [
            col for col in filtered_columns.copy() if exclude not in col
        ]

    return df[filtered_columns]


def clean_modelname(name: str, substring_to_replace: str, tablename: str):
    clean = (
        name.replace(f"{substring_to_replace}", "")
        .replace(".pickle", "")
        .replace("best", "")
        .replace(tablename, "")
        .replace("__", "_")
        .replace("__", "_")
    )
    clean = clean[:-1] if clean.endswith("_") else clean  # avoids ending in "_"
    return clean


def plot_forecasts(df, forecasts_by_stock, stock, top_k):
    fig, ax = plt.subplots(figsize=(12, 6))

    log_rets = df[f"{stock}_log_rets"]

    other_fcast = pd.DataFrame()
    for i in top_k:
        other_fcast = pd.concat(
            [other_fcast, subset_of_columns(forecasts_by_stock[stock], i)],
            axis=1,
            join="outer",
        )

    sb.lineplot(data=other_fcast, markers=True)
    sb.lineplot(data=log_rets, markers=True, label="Actual Returns", linewidth=1.1, c="black")

    plt.title(f"Forecasts Plot {stock} - Top {len(top_k)} forecasts", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Series", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    return fig


def plot_residuals(df, stock, show=True, return_fig=False):
  """
  Plots the residuals for a given stock.

  Args:
      df (pandas.DataFrame): The dataframe containing the data.
      stock (str): The name of the stock.
      show (bool, optional): Whether to display the plot directly. Defaults to True.
      return_fig (bool, optional): Whether to return the figure object. Defaults to False.

  Returns:
      plt.Figure (optional): The figure object containing the plot (if return_fig is True).
  """

  fig, ax = plt.subplots(figsize=(12, 6))  # Create the figure and axis

  sb.lineplot(data=df, markers=True, ax=ax)  # Plot residuals on the axis

  # Add title, labels, and formatting
  plt.title(f"Residuals Plot {stock}", fontsize=16)
  plt.xlabel("Date", fontsize=14)
  plt.ylabel("Values", fontsize=14)
  plt.xticks(rotation=45)
  plt.grid(True, linestyle="--", linewidth=0.5)
  plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels, title="Series", loc="upper left", bbox_to_anchor=(1, 1))

  plt.tight_layout()

  # Handle show and return options
  if show:
    plt.show()  # Display the plot
  if return_fig:
    return fig  # Return the figure object