# multi_label_email_classification_main

## Overview
In this project we implemnt two models for multilabeled email classification.
1. **Chained Model** - Uses a sequence of models where each model predicts one label and passes relevant information to the next.
2. **Hierarchical Model** - Uses a hierarchical structure to predict labels, considering dependencies between them.

The system processes textual email data, encodes categorical features, applies TF-IDF vectorization, and trains models using machine learning techniques.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SnehaMB/multi_label_email_classification_main
```

### 2. Set Up the Conda Environment
I'm using Anaconda environment named base.
Ensure you are using Conda's `base` environment.
```bash
conda activate base
```

### 3. Install Required Packages
1. From requirement.txt file
```bash
pip install -r requirements.txt
```
or

2.Manually install the dependencies:
```bash
pip install pandas deep-translator langdetect scikit-learn
```

## Project Structure
```
.
├── src
│   ├── config.py
│   ├── preprocessing.py
│   ├── models
│       ├── base_model.py
│       ├── chained_model.py
│       ├── hierarchial_model.py
│  
├── main.py
├── README.md
```

---
## Running the Project

### 1. Preprocess and Train Models
Run the `main.py` script to:
- Load and merge datasets
- Encode categorical labels
- Apply text preprocessing
- Train and evaluate both Chained and Hierarchical models

```bash
python main.py
```

### 2. Model Training & Evaluation
The models are trained using **RandomForestClassifier** with **MultiOutputClassifier**.

- **Chained Model**: `src.models.chained_model.ChainedModel`
- **Hierarchical Model**: `src.models.hierarchial_model.HierarchicalModel`

Each model is evaluated using **Accuracy Score, and F1-Score**.

## Authors
- **Sneha**
- **Sandra**

