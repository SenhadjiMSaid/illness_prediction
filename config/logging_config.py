import yaml

# Load YAML configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

print(config["app"]["name"])  # Output: Illness Prediction System
