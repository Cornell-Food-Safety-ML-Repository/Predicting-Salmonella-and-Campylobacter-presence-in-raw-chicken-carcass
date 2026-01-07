import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
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
    precip_mapping = {'None': 0, 'rain': 1, 'snow': 2, 'freezingrain': 3}
    for col in ['PrecipType_Day0', 'PrecipType_Day1', 'PrecipType_Day2', 'PrecipType_Day3']:
        df[col] = df[col].map(precip_mapping).fillna(0)
    df['Season_Number'] = df['CollectionDate'].apply(date_to_season).map({
        'spring': 1,
        'summer': 2,
        'autumn': 3,
        'winter': 4
    })
    df.drop(columns=['CollectionDate'], inplace=True)
    analysis_mapping = {'Positive': 1, 'Negative': 0}
    df['SalmonellaSPAnalysis'] = df['SalmonellaSPAnalysis'].map(analysis_mapping)
    df['CampylobacterAnalysis30ml'] = df['CampylobacterAnalysis30ml'].map(analysis_mapping)
    df.fillna(0, inplace=True)
    return df

def date_to_season(date):
    year = date.year
    seasons = {
        'spring': (pd.Timestamp(year=year, month=3, day=21), pd.Timestamp(year=year, month=6, day=20)),
        'summer': (pd.Timestamp(year=year, month=6, day=21), pd.Timestamp(year=year, month=9, day=22)),
        'autumn': (pd.Timestamp(year=year, month=9, day=23), pd.Timestamp(year=year, month=12, day=20)),
        'winter': (pd.Timestamp(year=year, month=12, day=21), pd.Timestamp(year=year, month=12, day=31)),
    }
    seasons['winter'] = (seasons['winter'][0].replace(year=year-1), seasons['winter'][1])
    for season, (start_date, end_date) in seasons.items():
        if start_date <= date <= end_date:
            return season
    return 'winter'

def analyze_label_distribution(y):
    label_distribution = y.value_counts(normalize=True) * 100
    print("Label distribution:\n", label_distribution)
    print("Number of labels:\n", y.value_counts())

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_params, resampling=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if resampling:
        ros = RandomOverSampler(random_state=42)
        X_train_scaled, y_train = ros.fit_resample(X_train_scaled, y_train)
        analyze_label_distribution(y_train)
    models = {
        'logistic_regression': LogisticRegression(max_iter=model_params['logistic_regression']['max_iter'], C=model_params['logistic_regression']['C']),
        'neural_network': MLPClassifier(max_iter=model_params['neural_network']['max_iter'], hidden_layer_sizes=model_params['neural_network']['hidden_layer_sizes']),
        'decision_tree': DecisionTreeClassifier(max_depth=model_params['decision_tree']['max_depth']),
        'svm': SVC(C=model_params['svm']['C'], kernel=model_params['svm']['kernel']),
        'knn': KNeighborsClassifier(n_neighbors=model_params['knn']['n_neighbors']),
        'gbm': GradientBoostingClassifier(n_estimators=model_params['gbm']['n_estimators'], learning_rate=model_params['gbm']['learning_rate'])
    }
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                             index=["Actual Negative", "Actual Positive"], 
                             columns=["Predicted Negative", "Predicted Positive"])
        print(f"{name} Model:\n"
              f"Confusion Matrix:\n{cm_df}\n"
              f"Accuracy: {accuracy:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Run machine learning models with configurable parameters.')
    parser.add_argument('filepath', type=str, help='Path to the dataset file.')
    parser.add_argument('--target', type=str, default='SalmonellaSPAnalysis', help='Target variable to predict.')
    parser.add_argument('--resampling', action='store_true', help='Whether to apply RandomOverSampler for class balancing.')

    # Model-specific parameters
    parser.add_argument('--lr_max_iter', type=int, default=1000, help='Maximum number of iterations for Logistic Regression.')
    parser.add_argument('--lr_C', type=float, default=1.0, help='Inverse of regularization strength for Logistic Regression.')
    parser.add_argument('--mlp_max_iter', type=int, default=1000, help='Maximum number of iterations for MLP Classifier.')
    parser.add_argument('--mlp_hidden_layers', type=int, nargs='+', default=[100], help='Number of neurons in the hidden layers for MLP Classifier.')
    parser.add_argument('--dt_max_depth', type=int, default=None, help='Maximum depth of the Decision Tree.')
    parser.add_argument('--svm_C', type=float, default=1.0, help='Regularization parameter for SVM.')
    parser.add_argument('--svm_kernel', type=str, default='rbf', help='Kernel type to be used in SVM.')
    parser.add_argument('--knn_n_neighbors', type=int, default=5, help='Number of neighbors to use for KNN.')
    parser.add_argument('--gbm_n_estimators', type=int, default=100, help='Number of boosting stages to perform for GBM.')
    parser.add_argument('--gbm_learning_rate', type=float, default=0.1, help='Learning rate for GBM.')

    args = parser.parse_args()

    model_params = {
        'logistic_regression': {'max_iter': args.lr_max_iter, 'C': args.lr_C},
        'neural_network': {'max_iter': args.mlp_max_iter, 'hidden_layer_sizes': tuple(args.mlp_hidden_layers)},
        'decision_tree': {'max_depth': args.dt_max_depth},
        'svm': {'C': args.svm_C, 'kernel': args.svm_kernel},
        'knn': {'n_neighbors': args.knn_n_neighbors},
        'gbm': {'n_estimators': args.gbm_n_estimators, 'learning_rate': args.gbm_learning_rate}
    }

    df = load_and_clean_data(args.filepath)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    analyze_label_distribution(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate_models(X_train, X_test, y_train, y_test, model_params, resampling=args.resampling)

    # Feature importance analysis with Gradient Boosting Machine
    gbm_model = GradientBoostingClassifier(n_estimators=model_params['gbm']['n_estimators'], learning_rate=model_params['gbm']['learning_rate']).fit(X_train, y_train)
    feature_importance_gbm = gbm_model.feature_importances_
    feature_importance_gbm_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance_gbm})
    print("Gradient Boosting Machine Feature Importance:\n", feature_importance_gbm_df.sort_values(by='Importance', ascending=False))

if __name__ == '__main__':
    main()
