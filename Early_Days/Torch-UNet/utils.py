import yaml

def load_settings(path: str):
    with open(path, "r") as file:
        settings = yaml.safe_load(file)
        return settings
