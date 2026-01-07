import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import SMOTE

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Columns to be dropped as specified
    columns_to_drop = [
        'EstablishmentID', 'EstablishmentNumber', 'EstablishmentName', 'State',
        'ProjectCode', 'ProjectName', 'FormID', 'SampleSource', 'SalmonellaSerotype',
        'SalmonellaPFGEPattern', 'SalmonellaAlleleCode', 'SalmonellaFSISNumber',
        'SalmonellaAMRResistanceProfile', 'CampylobacterAnalysis1ml', 'CampylobacterSpecies',
        'CampylobacterPFGEPattern', 'CampylobacterAlleleCode', 'CampyFSISNumber',
        'CampyAMRResistanceProfile', 'Weekday', 'WeatherDate_Day0', 'WeatherDate_Day1',
        'WeatherDate_Day2', 'WeatherDate_Day3'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Map 'Positive' to 1 and 'Negative' to 0
    analysis_mapping = {'Positive': 1, 'Negative': 0}
    df['SalmonellaSPAnalysis'] = df['SalmonellaSPAnalysis'].map(analysis_mapping)
    df['CampylobacterAnalysis30ml'] = df['CampylobacterAnalysis30ml'].map(analysis_mapping)

    # Fill NA values with 0
    df.fillna(0, inplace=True)
    
    # Convert categorical columns to strings
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = df[categorical_cols].astype(str)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Run machine learning models with LazyPredict.')
    parser.add_argument('filepath', type=str, help='Path to the dataset file.')
    parser.add_argument('--target', type=str, default='target', help='Target variable to predict.')

    args = parser.parse_args()

    df = load_and_clean_data(args.filepath)
    
    # Display the number of target=1
    target_count = df[args.target].sum()
    print(f"Number of target=1: {target_count}")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Apply preprocessing to the features
    X_preprocessed = preprocessor.fit_transform(X)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Using LazyPredict
    clf = LazyClassifier(predictions=True)
    models, predictions = clf.fit(X_train_smote, X_test, y_train_smote, y_test)

    # Display results
    print("Model Performance:")
    print(models)
    print("\nPredictions:")
    print(predictions)

if __name__ == '__main__':
    main()
