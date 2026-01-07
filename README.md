# *Salmonella* and *Campylobacter* in raw chicken carcass

🐔 This dataset contains meteorological and temporal data for raw chicken carcass samples tested for the presence of *Salmonella* and *Campylobacter*

📖 This dataset was sourced from “Dataset: Raw Poultry (Current)” at the USDA FSIS website (https://www.fsis.usda.gov/news-events/publications/raw-poultry-sampling). Our dataset includes only the samples categorized as "Animal-Chicken-Broiler / Young Chicken Carcass Rinse," as at the time of data acquisition in October 2023, testing results for other categories of raw poultry product were not merged. 

# Sample Analysis
## Overview

**Prediction task**:
- Classification for predicting the presence of Salmonella in the raw chicken carcass
- Classification for predicting the presence of Campylobacter in the raw chicken carcass

**Predictor and outcome variables**:
- The detailed description of metadata for predictor and outcome variables is accessible under the file name "SalCampChicken_Metadata.csv"
- The cleaned dataset is accessible under the file name "SalCampChicken_clean.csv"

**Evaluation metrics**:
- The classification model was evaluated on ROC AUC, sensitivity, specificity, and F1 score
- The specific packages for calculating these metrics are accessible in the model training script under the file name "Customize_script.py"


## Installation
### Dependencies
To use this script, you'll need Python 3.x and several libraries installed on your system:
- **Pandas**
- **Scikit-learn**
- **Imbalanced-learn**
- **LazyPredict**

You can install the necessary Python libraries using pip:
```bash
pip install -r requirements.txt
```

Alternatively, you can install them individually:
```bash
pip install pandas scikit-learn imbalanced-learn lazypredict
```
To get started with this project, follow these steps:

```bash
git clone[ https://github.com/FoodDatasets/Predicting-Salmonella-presence-in-raw-chicken-carcass.git
cd Predicting-Salmonella-presence-in-raw-chicken-carcass
```
## Supported Algorithms
-  Logistic Regression
-  Neural Network
-  Decision Tree
-  SVM
-  KNN
-  GBM
## Machine Learning Model Execution Guide
This script allows users to select different machine learning algorithms via command line parameters to train models and evaluate them on a specified dataset.
### Required Arguments
- ` --filepath`: Mandatory. The path to the dataset file.
- ` --model`: Optional. Choose the machine learning model. Default is 'logistic_regression'.
- ` --target`: Optional. Specify the target variable column. Default is 'target'.
### Optional Arguments
- ` --resampling`: Optional. Whether to apply RandomOverSampler for class balancing.
- ` --lr_C`: Optional. Regularization strength for logistic regression (inverse of lambda). Default is 1.0.
- ` --lr_max_iter`: Optional. Maximum iterations for logistic regression. Default is 100.
- ` --mlp_max_iter`: Optional. Maximum iterations for MLP classifier. Default is 200.
- ` --mlp_hidden_layers`: Optional. Number of neurons in the hidden layers for MLP Classifier.
- ` --dt_max_depth`: Optional. Maximum depth for the decision tree. Use 'None' for no limit. Default is 'None'.
- ` --svm_C`: Optional. Regularization parameter for SVM. Default is 1.0.
- ` --svm_kernel`: Optional. Kernel type for SVM. Default is 'rbf'.
- ` --knn_n_neighbors`: Optional. Number of neighbors for KNN. Default is 5.
- ` --gbm_n_estimators`: Optional. Number of boosting stages for GBM. Default is 100.
- ` --gbm_learning_rate`: Optional. Learning rate for GBM.
### Usage Example
Run the script from the command line, specifying the path to your dataset along with options to configure the model:
```bash
python Customize_script.py <path_to_dataset> --model <model_name> --target <target_column> [other options]
```
## Model Performance Results with Resampling Process

The following visualization and tables summarize the performance of different machine learning models after applying resampling.
### SalmonellaSPAnalysis
### Performance Comparison Chart

![Model Performance Comparison](Images/curve_new.png)

### Performance Table

| Algorithm           | Avg ROC AUC (Cross-validation) | Accuracy | Precision | Recall | F1 Score | ROC AUC (Test Set) |
|---------------------|-------------------------------|----------|-----------|--------|----------|--------------------|
| Neural Network       | 0.99                          | 0.91     | 0.96      | 0.95   | 0.95     | 0.57               |
| DecisionTree         | 0.97                          | 0.90     | 0.96      | 0.94   | 0.95     | 0.57               |
| GradientBoosting     | 0.97                          | 0.87     | 0.97      | 0.90   | 0.93     | 0.66               |
| KNN                 | 0.96                          | 0.85     | 0.95      | 0.89   | 0.92     | 0.55               |
| SVM                 | 0.95                          | 0.83     | 0.96      | 0.85   | 0.90     | 0.59               |
| LogisticRegression   | 0.70                          | 0.66     | 0.97      | 0.67   | 0.79     | 0.61               |



## Confusion Matrices

| Algorithm           | True Positive (TP) | False Positive (FP) | False Negative (FN) | True Negative (TN) |
|---------------------|--------------------|---------------------|---------------------|--------------------|
| Neural Network       | 5                  | 49                  | 40                  | 884                |
| DecisionTree         | 9                  | 60                  | 36                  | 873                |
| GradientBoosting     | 15                 | 94                  | 30                  | 839                |
| KNN                 | 6                  | 107                 | 39                  | 826                |
| SVM                 | 11                 | 136                 | 34                  | 797                |
| LogisticRegression   | 23                 | 311                 | 22                  | 622                |



![Model Performance Comparison](Images/output_roc1.png)
### CampylobacterAnalysis30ml

### Performance Comparison Chart
![Model Performance Comparison](Images/outputc.png)
### Performance Table

| Algorithm           | Avg ROC AUC (Cross-validation) | Accuracy | Precision | Recall | F1 Score | ROC AUC (Test Set) |
|---------------------|-------------------------------|----------|-----------|--------|----------|--------------------|
| Neural Network       | 0.85                          | 0.62     | 0.77      | 0.72   | 0.74     | 0.56               |
| DecisionTree         | 0.82                          | 0.61     | 0.75      | 0.72   | 0.74     | 0.49               |
| GradientBoosting     | 0.79                          | 0.63     | 0.78      | 0.72   | 0.75     | 0.56               |
| KNN                 | 0.72                          | 0.57     | 0.78      | 0.61   | 0.68     | 0.54               |
| SVM                 | 0.72                          | 0.58     | 0.78      | 0.62   | 0.69     | 0.54               |
| LogisticRegression   | 0.57                          | 0.53     | 0.77      | 0.54   | 0.64     | 0.54               |


### Confusion Matrices

| Algorithm           | True Positive (TP) | False Positive (FP) | False Negative (FN) | True Negative (TN) |
|---------------------|--------------------|---------------------|---------------------|--------------------|
| Neural Network       | 80                 | 210                 | 160                 | 528                |
| DecisionTree         | 62                 | 203                 | 178                 | 535                |
| GradientBoosting     | 88                 | 206                 | 152                 | 532                |
| KNN                 | 114                | 291                 | 126                 | 447                |
| SVM                 | 109                | 282                 | 131                 | 456                |
| LogisticRegression   | 120                | 338                 | 120                 | 400                |

![Model Performance Comparison](Images/output_roc2.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# LazyPredict Method
## Dependencies

- Python 3.7+
- pandas
- scikit-learn
- imbalanced-learn
- LazyPredict

``` bash
pip install pandas scikit-learn imbalanced-learn lazypredict
```
## Usage
1. Prepare your dataset file
2. Run the script with the dataset file and target column:
### Command-line Arguments
- `--file_path`: Path to the CSV file containing the dataset (required).
- `--target_label`:SalmonellaSPAnalysis or CampylobacterAnalysis30ml
```bash
python Lazy_script.py /path/to/SalCampChicken_clean.csv --target SalmonellaSPAnalysis( CampylobacterAnalysis30ml)
```

3. The script will output the number of positive cases for the target variable and evaluate various machine learning models.

## Example Output

### SalmonellaSPAnalysis
#### Model Performance Table

| Model                          | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken | Sensitivity | Specificity |
|--------------------------------|----------|-------------------|---------|----------|------------|-------------|-------------|
| LGBMClassifier                 | 0.95     | 0.55              | 0.55    | 0.94     | 0.21       | 0.11        | 0.99        |
| DummyClassifier                | 0.95     | 0.50              | 0.50    | 0.93     | 0.03       | 0.00        | 1.00        |
| ExtraTreesClassifier           | 0.94     | 0.52              | 0.52    | 0.93     | 0.86       | 0.04        | 0.99        |
| XGBClassifier                  | 0.94     | 0.56              | 0.56    | 0.93     | 0.33       | 0.13        | 0.98        |
| RandomForestClassifier         | 0.94     | 0.53              | 0.53    | 0.93     | 3.04       | 0.07        | 0.98        |
| BaggingClassifier              | 0.93     | 0.52              | 0.52    | 0.92     | 1.50       | 0.07        | 0.97        |
| LabelSpreading                 | 0.89     | 0.53              | 0.53    | 0.91     | 3.35       | 0.13        | 0.93        |
| LabelPropagation               | 0.89     | 0.53              | 0.53    | 0.91     | 2.94       | 0.13        | 0.93        |
| AdaBoostClassifier             | 0.88     | 0.65              | 0.65    | 0.90     | 1.53       | 0.40        | 0.90        |
| DecisionTreeClassifier         | 0.88     | 0.51              | 0.51    | 0.90     | 0.33       | 0.09        | 0.92        |
| ExtraTreeClassifier            | 0.86     | 0.55              | 0.55    | 0.89     | 0.04       | 0.20        | 0.89        |
| SVC                            | 0.86     | 0.56              | 0.56    | 0.89     | 1.87       | 0.22        | 0.89        |
| NuSVC                          | 0.85     | 0.57              | 0.57    | 0.88     | 2.70       | 0.27        | 0.88        |
| KNeighborsClassifier           | 0.74     | 0.57              | 0.57    | 0.81     | 0.05       | 0.38        | 0.76        |
| LogisticRegression             | 0.68     | 0.62              | 0.62    | 0.77     | 0.12       | 0.56        | 0.68        |
| SGDClassifier                  | 0.68     | 0.63              | 0.63    | 0.77     | 0.17       | 0.58        | 0.68        |
| CalibratedClassifierCV         | 0.67     | 0.62              | 0.62    | 0.77     | 6.89       | 0.56        | 0.68        |
| LinearSVC                      | 0.67     | 0.62              | 0.62    | 0.77     | 1.46       | 0.56        | 0.68        |
| RidgeClassifier                | 0.67     | 0.62              | 0.62    | 0.77     | 0.04       | 0.56        | 0.68        |
| RidgeClassifierCV              | 0.66     | 0.61              | 0.61    | 0.76     | 0.10       | 0.56        | 0.67        |
| LinearDiscriminantAnalysis     | 0.66     | 0.61              | 0.61    | 0.76     | 0.10       | 0.56        | 0.67        |
| Perceptron                     | 0.65     | 0.62              | 0.62    | 0.75     | 0.05       | 0.58        | 0.66        |
| PassiveAggressiveClassifier    | 0.64     | 0.57              | 0.57    | 0.74     | 0.07       | 0.49        | 0.65        |
| BernoulliNB                    | 0.63     | 0.57              | 0.57    | 0.73     | 0.03       | 0.51        | 0.63        |
| NearestCentroid                | 0.63     | 0.54              | 0.54    | 0.73     | 0.05       | 0.44        | 0.64        |
| QuadraticDiscriminantAnalysis  | 0.09     | 0.51              | 0.51    | 0.09     | 0.16       | 0.98        | 0.05        |
| GaussianNB                     | 0.07     | 0.50              | 0.50    | 0.06     | 0.04       | 0.98        | 0.03        |

#### Model Accuracy Comparison

![Model Accuracy Comparison](Images/output1sorted.png)

#### Model Comparison

![Model Comparison](Images/output1.png)
### CampylobacterAnalysis30ml
#### Model Performance Table

| Model                          | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken | Sensitivity | Specificity |
|--------------------------------|----------|-------------------|---------|----------|------------|-------------|-------------|
| DummyClassifier                | 0.75     | 0.50              | 0.50    | 0.65     | 0.02       | 0.00        | 1.00        |
| LGBMClassifier                 | 0.70     | 0.52              | 0.52    | 0.67     | 0.17       | 0.16        | 0.88        |
| RandomForestClassifier         | 0.70     | 0.52              | 0.52    | 0.66     | 1.75       | 0.16        | 0.87        |
| ExtraTreesClassifier           | 0.70     | 0.51              | 0.51    | 0.66     | 0.69       | 0.16        | 0.87        |
| XGBClassifier                  | 0.68     | 0.52              | 0.52    | 0.66     | 0.28       | 0.20        | 0.84        |
| BaggingClassifier              | 0.68     | 0.51              | 0.51    | 0.66     | 1.67       | 0.19        | 0.83        |
| AdaBoostClassifier             | 0.67     | 0.55              | 0.55    | 0.67     | 1.16       | 0.30        | 0.79        |
| NuSVC                          | 0.63     | 0.52              | 0.52    | 0.63     | 1.27       | 0.30        | 0.73        |
| DecisionTreeClassifier         | 0.62     | 0.50              | 0.50    | 0.62     | 0.25       | 0.28        | 0.73        |
| ExtraTreeClassifier            | 0.62     | 0.52              | 0.52    | 0.63     | 0.02       | 0.33        | 0.71        |
| LabelSpreading                 | 0.61     | 0.51              | 0.51    | 0.62     | 1.46       | 0.31        | 0.70        |
| LabelPropagation               | 0.61     | 0.51              | 0.51    | 0.62     | 1.23       | 0.30        | 0.71        |
| SVC                            | 0.58     | 0.54              | 0.54    | 0.61     | 1.14       | 0.46        | 0.62        |
| PassiveAggressiveClassifier    | 0.54     | 0.56              | 0.56    | 0.57     | 0.02       | 0.59        | 0.52        |
| RidgeClassifierCV              | 0.54     | 0.53              | 0.53    | 0.58     | 0.13       | 0.51        | 0.55        |
| CalibratedClassifierCV         | 0.54     | 0.52              | 0.52    | 0.57     | 7.09       | 0.48        | 0.56        |
| Perceptron                     | 0.54     | 0.50              | 0.50    | 0.57     | 0.02       | 0.43        | 0.57        |
| LogisticRegression             | 0.53     | 0.52              | 0.52    | 0.56     | 0.06       | 0.50        | 0.54        |
| RidgeClassifier                | 0.53     | 0.52              | 0.52    | 0.57     | 0.03       | 0.48        | 0.55        |
| SGDClassifier                  | 0.53     | 0.52              | 0.52    | 0.56     | 0.07       | 0.49        | 0.54        |
| LinearDiscriminantAnalysis     | 0.53     | 0.52              | 0.52    | 0.57     | 0.10       | 0.48        | 0.55        |
| LinearSVC                      | 0.53     | 0.51              | 0.51    | 0.56     | 0.73       | 0.47        | 0.55        |
| KNeighborsClassifier           | 0.51     | 0.52              | 0.52    | 0.54     | 0.10       | 0.55        | 0.49        |
| NearestCentroid                | 0.51     | 0.52              | 0.52    | 0.55     | 0.02       | 0.52        | 0.51        |
| BernoulliNB                    | 0.48     | 0.50              | 0.50    | 0.51     | 0.03       | 0.53        | 0.46        |
| GaussianNB                     | 0.26     | 0.51              | 0.51    | 0.14     | 0.02       | 1.00        | 0.03        |
| QuadraticDiscriminantAnalysis  | 0.26     | 0.51              | 0.51    | 0.13     | 0.09       | 1.00        | 0.02        |


#### Model Accuracy Comparison

![Model Accuracy Comparison](Images/output2new.png)

#### Model Comparison

![Model Comparison](Images/output2.png)
