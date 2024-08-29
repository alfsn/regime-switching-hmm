import os
import yaml


def get_params():
    with open(os.path.join("..", "notebooks", "current_run.txt"), "r") as file:
        config_file = file.read()

    if not config_file.endswith(".yml"):
        config_file = config_file + ".yml"

    with open(os.path.join("..", "notebooks", config_file), "r") as file:
        yaml_data = yaml.safe_load(file)

    tablename_prefix = yaml_data["tablename_prefix"]
    index = yaml_data["index"]
    yaml_data["tablename"] = f"""{tablename_prefix}_{index}"""

    locallist = []
    foreignlist = []
    stockslist = []
    for stock, lista in yaml_data["stocksdict"].items():
        local = lista[1]
        ext = lista[0]
        locallist.append(local)  # only local currency items
        foreignlist.append(ext)  # only ADRs

    # only stock tickers - excludes index
    stockslist = foreignlist.copy() + locallist.copy()

    yaml_data["foreignlist"] = foreignlist.copy()
    yaml_data["locallist"] = locallist.copy()
    yaml_data["stockslist"] = stockslist.copy()

    # all downloadable tickers
    yaml_data["tickerlist"] = [yaml_data["index"]] + yaml_data["stockslist"].copy()

    yaml_data["synth_index"] = yaml_data["index"].replace("^", "") + "_FX"
    # all assets - includes synthethic index
    yaml_data["assetlist"] = [yaml_data["synth_index"]] + yaml_data["tickerlist"].copy()

    yaml_data["dataroute"] = os.path.join("..", "data", yaml_data["tablename"])
    yaml_data["dumproute"] = os.path.join("..", "dump", yaml_data["tablename"])
    yaml_data["resultsroute"] = os.path.join("..", "results", yaml_data["tablename"])
    yaml_data["graphsroute"] = os.path.join("..", "graphs", yaml_data["tablename"])
    yaml_data["descriptivegraphsroute"] = os.path.join(
        yaml_data["graphsroute"], "descriptive"
    )
    yaml_data["dmroute"] = os.path.join(yaml_data["graphsroute"], "DM")
    yaml_data["gwroute"] = os.path.join(yaml_data["graphsroute"], "GW")

    yaml_data["forecastsgraphsroute"] = os.path.join(
        yaml_data["graphsroute"], "forecasts"
    )
    yaml_data["residualsgraphsroute"] = os.path.join(
        yaml_data["graphsroute"], "residuals"
    )

    model_graphs = []
    for model in yaml_data["model_list"]:
        for dtype in ["forecasts", "residuals"]:
            yaml_data[f"{model}_{dtype}"] = os.path.join(
                yaml_data[f"{dtype}graphsroute"], f"{model}"
            )
            model_graphs.append(yaml_data[f"{model}_{dtype}"])

    yaml_data["directories"] = [
        yaml_data["dataroute"],
        yaml_data["dumproute"],
        yaml_data["resultsroute"],
        yaml_data["graphsroute"],
        yaml_data["descriptivegraphsroute"],
        yaml_data["dmroute"],
        yaml_data["gwroute"],
    ] + model_graphs

    return yaml_data
