# Defining global configurations for paths, model parameters, etc.
DATA_PATH_1 = "data/AppGallery.csv"
DATA_PATH_2 = "data/Purchasing.csv"
MERGED_DATA_FILE = "data/merged_dataset.csv"
ENCODED_DATA_FILE = "data/encoded_dataset.csv"
CLEANED_DATA_FILE = "data/cleaned_dataset.csv"
LABEL_COLUMNS = ["Type 2", "Type 3", "Type 4"]  # Target labels
INTERACTION_CONTENT = "Interaction content"
MODEL_SAVE_PATH = "models/"
CHAINED_LABELS = [
    ["Type 2"],
    ["Type 3"],
    ["Type 4"],
    ["Type 2", "Type 3"],
    ["Type 3", "Type 4"],
    ["Type 2", "Type 3", "Type 4"],
]

HIERARCHICAL_LABELS = [
    ["Type 1"],
    ["Type 2"],
    ["Type 3"],
    ["Type 4"],
    ["Type 2", "Type 3"],
    ["Type 1", "Type 3"],
    ["Type 1", "Type 2"],
    ["Type 1", "Type 2", "Type 3"],
    ["Type 1", "Type 2", "Type 3", "Type 4"],
    ["Type 1", "Type 4"],
    ["Type 2", "Type 4"],
    ["Type 3", "Type 4"],
    [
        "Type 1",
        "Type 2",
        "Type 4",
    ],
    [
        "Type 1",
        "Type 3",
        "Type 4",
    ],
]
