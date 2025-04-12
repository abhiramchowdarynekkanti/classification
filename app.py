# performing all algo without tuning 

# Full pipeline: Data cleaning, feature engineering, training models, evaluating, and printing results

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
import numpy as np
df = pd.read_csv("/content/titanic_data.csv")
# Step 1: Data Cleaning
df_full = df.copy()

# Drop irrelevant columns
df_full.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], inplace=True)

# Handle missing numeric values
df_full['age'] = SimpleImputer(strategy='median').fit_transform(df_full[['age']])

# Fill categorical NaNs with the most frequent value
df_full['embarked'] = df_full['embarked'].fillna(df_full['embarked'].mode()[0])

# Encode categorical variables
le = LabelEncoder()
df_full['sex'] = le.fit_transform(df_full['sex'])  # male=1, female=0
df_full['embarked'] = le.fit_transform(df_full['embarked'])

# Create new feature: family_size = sibsp + parch
df_full['family_size'] = df_full['sibsp'] + df_full['parch']
df_full.drop(columns=['sibsp', 'parch'], inplace=True)

# Step 2: Train-test split
X = df_full.drop(columns='survived')
y = df_full['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define and train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Linear Regression': LinearRegression()  # for comparison
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Handle linear regression separately (continuous output)
    if name == 'Linear Regression':
        y_pred_class = (y_pred >= 0.5).astype(int)
        r2 = r2_score(y_test, y_pred)
    else:
        y_pred_class = y_pred
        r2 = r2_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None

    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred_class), 4),
        'Precision': round(precision_score(y_test, y_pred_class), 4),
        'Recall': round(recall_score(y_test, y_pred_class), 4),
        'F1 Score': round(f1_score(y_test, y_pred_class), 4),
        'R² Score': round(r2, 4) if r2 is not None else '—'
    })

# Create a DataFrame for results
results_df = pd.DataFrame(results)

# Sort and display the results
results_df_sorted = results_df.sort_values(by='F1 Score', ascending=False)
results_df_sorted.reset_index(drop=True, inplace=True)
results_df_sorted






# performing all algo with tuning 

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

# Load dataset
df = pd.read_csv("/content/titanic_data.csv")

# Data cleaning and feature engineering
df_full = df.copy()
df_full.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], inplace=True)
df_full['age'] = SimpleImputer(strategy='median').fit_transform(df_full[['age']])
df_full['embarked'] = df_full['embarked'].fillna(df_full['embarked'].mode()[0])

le = LabelEncoder()
df_full['sex'] = le.fit_transform(df_full['sex'])  # male=1, female=0
df_full['embarked'] = le.fit_transform(df_full['embarked'])

df_full['family_size'] = df_full['sibsp'] + df_full['parch']
df_full.drop(columns=['sibsp', 'parch'], inplace=True)

# Train-test split
X = df_full.drop(columns='survived')
y = df_full['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and parameter grids (reduced for speed)
models_params = {
    'Logistic Regression': (
        LogisticRegression(max_iter=1000),
        {'C': [0.1, 1, 10]}
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {'max_depth': [5, 10], 'min_samples_split': [2, 5]}
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
    ),
    'SVM': (
        SVC(probability=True),
        {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(),
        {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3]}
    )
}

# Train with hyperparameter tuning
results = []

for name, (model, params) in models_params.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(model, params, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    try:
        r2 = r2_score(y_test, best_model.predict_proba(X_test)[:, 1])
    except:
        r2 = None

    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1 Score': round(f1_score(y_test, y_pred), 4),
        'R² Score': round(r2, 4) if r2 is not None else '—'
    })

# Display results
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
print("\n✅ Final Tuned Model Performance:")
print(results_df_sorted)
