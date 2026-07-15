import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MAPPING_PATH = BASE_DIR / "models" / "disease_mapping.json"

with open(MAPPING_PATH) as f:
    disease_mapping = json.load(f)

disease_names = list(disease_mapping.values())