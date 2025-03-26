import os
import sys
import warnings

# Ignore the warnings

warnings.filterwarnings("ignore")

# add the src directory to the system path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from src.config import CLEANED_DATA_FILE
from src.models.chained_model import ChainedModel
from src.models.hierarchial_model import HierarchicalModel
from src.preprocessing import (
    de_duplication,
    encode_labels,
    load_and_merge_datasets,
    preprocess_text,
)

if __name__ == "__main__":
    # Step 1: Loading and merge the atasets of AppGallery and Purchasing
    load_and_merge_datasets()

    # Step 2: Encode labels in the merged dataset
    encode_labels()

    # Step 3: preprocessing the text
    preprocess_text()

    # Step 4: de duplication by loading the data
    
    df = pd.read_csv(CLEANED_DATA_FILE)
    df = de_duplication(df)

    # fival preprocessed data saved
    
    df.to_csv("final_processed_data.csv", index=False)
    print("Final processed data saved as 'final_processed_data.csv'")

    # loading the preprocessed data
    
    print("Loading preprocessed data...")
    df = pd.read_csv("final_processed_data.csv")

    # Step 5: for each model 
    X = df.drop(columns=["Type 1", "Type 2", "Type 3", "Type 4"])

 
    # Chained Model targets and target variables
    
    y_chained = df[["Type 2", "Type 3", "Type 4"]]
    
    # Hierarchical Model targets
    
    y_hierarchical = df[["Type 1", "Type 2", "Type 3", "Type 4"]]

    # Handle categorical columns like 'Mailbox'
    
    label_encoder = LabelEncoder()
    if "Mailbox" in X.columns:
        X["Mailbox"] = label_encoder.fit_transform(X["Mailbox"])

    # Handle text columns with TF-IDF vectorization
    
    if "Ticket Summary" in X.columns:
        X["Ticket Summary"] = X["Ticket Summary"].fillna("")

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X["Ticket Summary"])
    X = X.drop(columns=["Ticket Summary"])  # Drop the column after this
    X = pd.concat([X, pd.DataFrame(X_tfidf.toarray())], axis=1)

    #  non-numeric columns label encodings can be done
    
    for column in X.columns:
        if X[column].dtype == "object":
            X[column] = label_encoder.fit_transform(X[column].astype(str))

    # Converting column names to strings
    
    X.columns = X.columns.astype(str)

    # missing values can be imputed
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Step 6: dataset can be split seperately for each model
    
    X_train, X_test, y_train_chained, y_test_chained = train_test_split(
        X, y_chained, test_size=0.3, random_state=42
    )
    X_train, X_test, y_train_hierarchical, y_test_hierarchical = train_test_split(
        X, y_hierarchical, test_size=0.3, random_state=42
    )

    # consistency ensured
    
    if len(X_train) != len(y_train_chained) or len(X_train) != len(
        y_train_hierarchical
    ):
        raise ValueError("Mismatch between feature and target dataset lengths!")

    # Train & Evaluation of Chained Model
    
    print("\n= Training the Chained Model =")
    chained_model = ChainedModel()
    chained_model.train(X_train, y_train_chained)

    print("Predicting with the Chained Model...")
    y_pred_chained = chained_model.predict(X_test)

    print("Evaluating the Chained Model...")
    chained_model.print_results(y_test_chained, y_pred_chained)

    # Train & Evaluation of the Hierarchical Model
    
    print("\n= Training the Hierarchical Model =")
    hierarchical_model = HierarchicalModel()
    hierarchical_model.train(X_train, y_train_hierarchical)

    print("Predicting with the Hierarchical Model...")
    y_pred_hierarchical = hierarchical_model.predict(X_test)

    print("Evaluating the Hierarchical Model...")
    hierarchical_model.print_results(y_test_hierarchical, y_pred_hierarchical)
