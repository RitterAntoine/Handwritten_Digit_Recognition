import json

def load_config(config_path='config.json'):
    """Loads the configuration from a JSON file."""
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config