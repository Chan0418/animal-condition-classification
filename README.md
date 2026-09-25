# animal-condition-classification

## Project Overview

This project uses machine learning to predict whether an animal condition is classified as dangerous based on the animal type and reported symptoms.

The project compares three classification models:

- Logistic Regression
- Linear Support Vector Machine
- Random Forest

## Research Question

Can animal type and reported symptoms predict whether an animal condition is dangerous?

## Dataset

The dataset comes from the Kaggle dataset:

[`gracehephzibahm/animal-disease`](https://www.kaggle.com/datasets/gracehephzibahm/animal-disease)

The features include animal type and five symptom fields
The target variable is `Dangerous`, containing `yes` or `no` labels.

## Methods

The notebook includes:

1. Data loading and inspection
2. Data normalization, preparation and cleaning
3. Exploratory data analysis
4. Encoding for categorical variables
5. Machine-learning pipelines
6. Stratified 5-fold cross-validation
7. Model comparison using accuracy, precision, recall, F1-score, and ROC-AUC
8. Test-set evaluation
9. Confusion matrices

The final model is selected using cross-validated F1-score because the dataset is highly imbalanced.

## How to Run

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

Then open and run:

`animal_disease_project.ipynb`

The notebook can load the dataset through KaggleHub. Alternatively, place a file named `data.csv` in the same folder as the notebook.

## Output Files

When the notebook runs, it creates:

- `clean_data.csv`
- `model_comparison.csv`
- `model_comparison_test.csv`

## Limitations

The dataset is strongly imbalanced, with many more dangerous cases than non-dangerous cases. The minority class is small, so evaluation results may be unstable. The dataset may also contain duplicate or very similar records. Therefore, high accuracy should not automatically be interpreted as strong real-world performance. The model identifies patterns in this dataset but does not establish medical or veterinary causation.
