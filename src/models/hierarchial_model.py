from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.base_model import BaseModel
from src.config import HIERARCHICAL_LABELS


class HierarchicalModel(BaseModel):
    def __init__(self):
        self.models = {}

    def train(self, X_train, y_train):

        # Train separate models for each hierarchy (label set).
        for label_set in HIERARCHICAL_LABELS:
            print(f"Training model for labels: {label_set}")
            # Use label_set[0] to access the correct column
            y_train_subset = y_train[label_set[0]]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train_subset)
            self.models[tuple(label_set)] = model
        print("Training completed for all hierarchical models.")

    def predict(self, X_test):

        # Predict on the test data for each label.
        predictions = {}
        for label_set, model in self.models.items():
            print(f"Predicting for labels: {label_set}")
            predictions[label_set] = model.predict(X_test)
        return predictions

    def print_results(self, y_test, y_pred):
        # Evaluate the models' performance by calculating overall accuracy and F1 score.
        all_y_test = []
        all_y_pred = []

        for label_set in HIERARCHICAL_LABELS:
            print(f"Results for labels {label_set}:")
            # Use label_set[0] to get the correct column in y_test
            y_test_subset = y_test[label_set[0]]
            y_pred_subset = y_pred[tuple(label_set)]

            # Append the values to the overall lists for accuracy and F1 score calculation
            all_y_test.extend(
                y_test_subset.values.flatten()
            )  # Flatten the values for F1 score calculation
            all_y_pred.extend(y_pred_subset.flatten())

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(all_y_test, all_y_pred)

        # Calculate overall F1 score
        overall_f1 = f1_score(all_y_test, all_y_pred, average="micro")

        print(f"Overall Accuracy: {overall_accuracy:.2f}")
        print(f"Overall F1 score: {overall_f1:.2f}")
