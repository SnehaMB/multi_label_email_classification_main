import re

import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import config
from config import (
    CLEANED_DATA_FILE,
    DATA_PATH_1,
    DATA_PATH_2,
    ENCODED_DATA_FILE,
    INTERACTION_CONTENT,
    LABEL_COLUMNS,
    MERGED_DATA_FILE,
)


# Loading and merging two datasets
def load_and_merge_datasets():
    df1 = pd.read_csv(DATA_PATH_1)
    df2 = pd.read_csv(DATA_PATH_2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(MERGED_DATA_FILE, index=False)
    print(f"Datasets merged and saved as {MERGED_DATA_FILE}")


# Encoding labels
def encode_labels():
    df = pd.read_csv(MERGED_DATA_FILE)
    label_encoders = {}
    for col in LABEL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    df.to_csv(ENCODED_DATA_FILE, index=False)
    print(f"Label Encoding done and saved as {ENCODED_DATA_FILE}")


# Transilating to english language
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source="auto", target="en").translate(text)
    except:
        return text  # Return original if translation fails
    return text


# Cleaning content, removing special data that affect modeling
def clean_text(text):
    """Clean the email text by removing HTML tags, extra spaces, and trimming leading/trailing spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = text.strip()  # Strip leading/trailing spaces
    return text


# Preprocessing
def preprocess_text():
    df = pd.read_csv(ENCODED_DATA_FILE)

    # Define noise patterns to remove unwanted text
    noise_patterns = [
        "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])",
        "(january|february|march|april|may|june|july|august|september|october|november|december)",
        "\d{2}(:|.)\d{2}",
        "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        "dear ((customer)|(user))",
        "(hello)|(hallo)|(hi )|(hi there)",
        "thank you for contacting us",
        "we apologize for the inconvenience",
        "\d+",
        "[^0-9a-zA-Z]+",
        "(\s|^).(\s|$)",
    ]

    # Preprocess the 'Interaction content' text column
    df["cleaned_text"] = df["Interaction content"].str.lower()
    for pattern in noise_patterns:
        df["cleaned_text"] = (
            df["cleaned_text"].str.replace(pattern, " ", regex=True).str.strip()
        )

    # Translate non-English text to English
    df["cleaned_text"] = df["cleaned_text"].apply(translate_to_english)

    # Handle email addresses by replacing them with '<EMAIL>'
    df["cleaned_text"] = df["cleaned_text"].apply(
        lambda x: re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "<EMAIL>", str(x)
        )
    )

    # Convert the text data into numerical form using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

    # Convert the TF-IDF result into a DataFrame and concatenate with the original dataset
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
    )
    df = pd.concat([df, tfidf_df], axis=1)

    # Now the df contains the numerical representation of the text and can be used for model training
    df.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"Text preprocessing completed and saved as {CLEANED_DATA_FILE}")


# Deleting duplicated entries
def de_duplication(data: pd.DataFrame):
    """Remove duplicate interaction content based on predefined templates for customer support emails."""
    cu_template = {
        "english": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?",
        ],
        "german": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
            r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?",
        ],
        # Add more languages as needed
    }

    cu_pattern = "|".join(
        [f"({x})" for sublist in cu_template.values() for x in sublist]
    )

    # Email split template
    pattern_1 = r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = r"(On.{30,60}wrote:)"
    pattern_3 = r"(Re\s?:|RE\s?:)"
    pattern_4 = r"(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = r"(\s?\*\*\*\*\*\(PHONE\))*$"
    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    # Processing ticket data
    tickets = data["Ticket id"].value_counts()

    for t in tickets.index:
        df = data.loc[data["Ticket id"] == t, :]
        ic_set = set()
        ic_deduplicated = []
        for ic in df[config.INTERACTION_CONTENT]:
            if isinstance(ic, str):
                ic_r = re.split(split_pattern, ic)
                ic_r = [i for i in ic_r if i is not None]
                ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]
                ic_r = [re.sub(cu_pattern, "", i.strip()) for i in ic_r]

                ic_current = [i + "\n" for i in ic_r if len(i) > 0 and i not in ic_set]
                ic_set.update(ic_current)
                ic_deduplicated.append(" ".join(ic_current))
            else:
                ic_deduplicated.append("")

        data.loc[data["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated

    data[config.INTERACTION_CONTENT] = data["ic_deduplicated"]
    data = data.drop(columns=["ic_deduplicated"])

    return data
