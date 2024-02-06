import os
import yaml

def get_params():
    with open(os.path.join("..", "notebooks", "current_run.txt"), 'r') as file:
        config_file = file.read()

    if not config_file.endswith('.yml'):
         config_file = config_file + '.yml'

    with open(os.path.join("..", "notebooks", config_file), 'r') as file:
        yaml_data = yaml.safe_load(file)

    tablename_prefix = yaml_data['tablename_prefix']
    index = yaml_data['index']
    yaml_data['tablename'] = f"""{tablename_prefix}_{index}"""
    
    yaml_data['stockslist']=[]
    # only stock tickers - excludes index
    for stock, lista in yaml_data["stocksdict"].items():
        local=lista[1] 
        ext=lista[0]
        yaml_data['stockslist'].append(local)
        yaml_data['stockslist'].append(ext)
    
    # all downloadable tickers
    yaml_data['tickerlist'] = [yaml_data['index']]+yaml_data['stockslist'].copy()
    # all assets - includes synthethic index
    yaml_data['assetlist'] = [yaml_data['index'], "USD_"+yaml_data['index']]+yaml_data['stockslist'].copy()
    
    return yaml_data
