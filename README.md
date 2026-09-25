# animal_condition_project README.md

## Project Overview

This project uses machine learning to predict whether an animal health condition is classified as dangerous based on the animal type and reported symptoms.

The project compares three classification models:

- Logistic Regression
- Linear Support Vector Machine
- Random Forest

## Research Question

Can animal type and reported symptoms be used to predict whether a condition is classified as dangerous?

## Dataset

The project uses the [Animal Disease dataset from Kaggle](https://www.kaggle.com/datasets/gracehephzibahm/animal-disease).

The dataset contains:

- Animal type
- Five symptom fields
- A binary `Dangerous` target variable with `yes` and `no` labels

After cleaning and removing invalid or duplicate records, the notebook uses 839 records for analysis.

## Methods

The notebook follows these steps:

1. Load and inspect the dataset
2. Standardize animal and symptom text
3. Apply selected corrections for known spelling and naming variations
4. Explore the target distribution and common symptoms
5. Split the data into training and test sets
6. Apply one-hot encoding through leakage-safe pipelines
7. Compare models using stratified five-fold cross-validation
8. Evaluate the models on a held-out test set
9. Display confusion matrices and classification reports

The primary model-selection metric is F1-score because the dataset is highly imbalanced. Accuracy, precision, recall, and ROC-AUC are also reported.

## How to Run

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

Then open and run:

```text
animal_disease_project.ipynb
```

The notebook first looks for `data.csv` in the current folder. If it is not found, it attempts to load the dataset using KaggleHub. You can also place the downloaded `data.csv` file beside the notebook.

## Output Files

Running the notebook creates:

- `clean_data.csv` — the cleaned dataset
- `model_comparison.csv` — cross-validation results
- `model_comparison_test.csv` — held-out test-set results

## Results and Interpretation

The Linear SVM achieved the strongest cross-validated F1-score and was selected as the final model. It also classified all records in the held-out test set correctly.

This result should be interpreted carefully because the dataset contains 819 dangerous cases and only 20 non-dangerous cases. The test set therefore contains only a small number of non-dangerous examples. A larger and more balanced dataset would be needed to determine how well the model generalizes to new cases.

## Limitations

- The target classes are strongly imbalanced.
- The minority class contains relatively few observations.
- Duplicate or highly similar records may lead to optimistic results.
- The dataset may not represent real-world veterinary cases.
- The model learns associations from the dataset and does not establish medical causation.

## Conclusion

This project demonstrates a complete classification workflow for animal-disease risk prediction, including data cleaning, exploratory analysis, model comparison, and test-set evaluation. The results provide a useful starting point, but future work should use a larger, more balanced, and independently collected dataset.

## Disclaimer

This project is for educational purposes only. It is not a veterinary diagnostic tool, and its predictions should not be used to make animal-health decisions.
