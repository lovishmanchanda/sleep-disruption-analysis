import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_and_save():
    print("Loading regression data...")
    df = pd.read_csv('data/df_regression_v2.csv')
    
    X = df.drop(columns=['sleep_quality_score'])
    y = df['sleep_quality_score']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Scaling data...")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    print("Training Random Forest Regressor with Grid Search...")
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")
    
    score = best_rf.score(X_test_scaled, y_test)
    print(f"Test R^2 Score: {score:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_rf, 'models/regression_model.joblib')
    joblib.dump(scaler, 'models/regression_scaler.joblib')
    joblib.dump(list(X.columns), 'models/regression_features.joblib')
    print("Regression model updated and saved.")

if __name__ == "__main__":
    train_and_save()
