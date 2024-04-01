import os
import pickle

def generate_columns(stock: str, contains_vol: bool, contains_USD: bool):
    """Devuelve una lista con los nombres de columnas para distintas especificaciones"""
    columns = []
    columns.append(f"{stock}_log_rets")

    if contains_vol:
        columns.append(f"{stock}_gk_vol")

    if contains_USD:
        columns.append(f"USD_log_rets")
        columns.append(f"USD_gk_vol")

    return columns


def save_as_pickle(data, resultsroute:str, model_type:str, tablename:str, criterion: str, type_save: str):    
    with open(
        os.path.join(
            resultsroute,
            f"""{model_type}_{tablename}_{criterion}_best_{type_save}.pickle""",
        ),
        "wb",
    ) as output_file:
        pickle.dump(data, output_file)