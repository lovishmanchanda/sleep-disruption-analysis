import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save():
    print("Loading data for classification...")
    df = pd.read_csv('data/df_classification.csv')
    
    # Drop features that the user wouldn't know when using the app
    features_to_drop = ['attention_fragmentation', 'sleep_quality_score', 'sleep_efficiency_score']
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['attention_fragmentation']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling data...")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print("Training Random Forest with Grid Search for Best Accuracy...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_rf = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")
    
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    score = best_rf.score(X_test_scaled, y_test)
    print(f"Test Accuracy (without cheat features): {score:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_rf, 'models/rf_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(list(X.columns), 'models/features.joblib')
    print("Classification model updated and saved.")

if __name__ == "__main__":
    train_and_save()
