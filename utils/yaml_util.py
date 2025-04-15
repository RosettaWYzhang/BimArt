import yaml

def load_and_merge_yaml_recur(base_file, derived_content):
    with open(base_file, 'r') as base:
        print("loading base file in recur: ", base_file)
        base_content = yaml.safe_load(base)
    merge_dicts(base_content, derived_content) # merge first before recursion
    if base_content["inherit"] != None: 
        base_content = load_and_merge_yaml_recur(base_content["inherit"], base_content)
    return base_content


def merge_dicts(base, derived):
    for key, value in derived.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_dicts(base[key], value)
        else:
            if key == "inherit":
                continue # do not merge inherit key
            else:
                print("update key %s to value %s: " %(key, str(value))) 
                base[key] = value


def load_yaml(file_name):
    '''TODO: Make it recursive function to load yaml file and its base file
    '''
    with open(file_name, 'r') as curr_config:
        content = yaml.safe_load(curr_config)

    if content["inherit"] == None:
        return content
    else:
        return load_and_merge_yaml_recur(content["inherit"], content)

def save_yaml(file_name, content):
    with open(file_name, 'w') as f:
        yaml.dump(content, f, default_flow_style=False)