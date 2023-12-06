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
    start_train = yaml_data['start_train']
    end_test = yaml_data['end_test']
    yaml_data['tablename'] = f"""{tablename_prefix}_{index}_{start_train.replace("-","_")}__{end_test.replace("-","_")}"""
    
    return yaml_data
