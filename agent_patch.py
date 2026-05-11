import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from joblib import dump

# Load the data
data = pd.read_parquet('/home/marno/Documents/NCF/Machine_Learning-20260305T162318Z-3-001/Machine_Learning/Project/train_split.parquet')

# Drop the 'label' column to get features
X = data.drop(columns=['label'])
y = data['label']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Set up the RandomizedSearchCV
rf = RandomForestClassifier()
random_search = RandomizedSearchCV(rf, param_grid, n_iter=100, cv=5, scoring='accuracy', random_state=42)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Save the trained model to /home/marno/Documents/NCF/Machine_Learning-20260305T162318Z-3-001/Machine_Learning/Project/models/specialist_20260506_002421.joblib
dump(best_model, '/home/marno/Documents/NCF/Machine_Learning-20260305T162318Z-3-001/Machine_Learning/Project/models/specialist_20260506_002421.joblib')

print("Best Parameters:", best_params)