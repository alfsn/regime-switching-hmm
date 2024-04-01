import os
import pickle
import pandas as pd


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
    name.replace(f"{substring_to_replace}", ""
                 ).replace(".pickle", ""
                           ).replace("best", ""
                                     ).replace(tablename, ""
                                               ).replace("__", "_"
                                                         ).replace("__", "_")
    return name
