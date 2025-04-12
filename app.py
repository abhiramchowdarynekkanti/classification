import streamlit as st
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

# ---- Streamlit UI ----
st.set_page_config(page_title="🚢  ML Dashboard", layout="centered")
st.title("🚢 Classification ")
st.write("This app trains multiple ML models on the Titanic dataset and shows their performance.")

# ---- Load Data ----
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# ---- File Upload ----
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("🗃️ Raw Data Preview")
    st.dataframe(df.head())
else:
    st.warning("Please upload the  dataset to continue.")

# ---- Preprocess ----
def preprocess(df):
    df = df.copy()
    df.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], inplace=True)
    df['age'] = SimpleImputer(strategy='median').fit_transform(df[['age']])
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['embarked'] = le.fit_transform(df['embarked'])
    df['family_size'] = df['sibsp'] + df['parch']
    df.drop(columns=['sibsp', 'parch'], inplace=True)
    return df

if uploaded_file is not None:
    df_clean = preprocess(df)

    # ---- Train-test Split ----
    X = df_clean.drop(columns='survived')
    y = df_clean['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- Models ----
    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Gradient Boosting': GradientBoostingClassifier()
    }

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
            {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        ),
        'SVM': (
            SVC(probability=True),
            {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(),
            {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        )
    }

    # ---- Select Mode ----
    mode = st.radio("🛠️ Select Training Mode", ["Without Tuning", "With Tuning"])

    # ---- Training & Evaluation ----
    results = []

    with st.spinner(f"Training models ({mode})..."):
        if mode == "Without Tuning":
            for name, model in base_models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                try:
                    r2 = r2_score(y_test, model.predict_proba(X_test)[:, 1])
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

        else:
            for name, (model, params) in models_params.items():
                grid = GridSearchCV(model, params, cv=3, scoring='f1', n_jobs=-1)
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

    # ---- Display Results ----
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(results_df)

    best_model_name = results_df.iloc[0]['Model']
    st.success(f"🎉 Best Model ({mode}): **{best_model_name}**")
        # ---- Display Results ----
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(results_df)

    best_model_name = results_df.iloc[0]['Model']
    st.success(f"🎉 Best Model ({mode}): **{best_model_name}**")

    # ---- Use Best Model to Predict ----
    st.subheader("🔮 Predictions by Best Model")

    if mode == "Without Tuning":
        best_model = base_models[best_model_name]
        best_model.fit(X_train, y_train)
    else:
        model, params = models_params[best_model_name]
        grid = GridSearchCV(model, params, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    predictions = best_model.predict(X_test)

    pred_df = X_test.copy()
    pred_df['Actual'] = y_test.values
    pred_df['Predicted'] = predictions
    pred_df['Predicted'] = pred_df['Predicted'].map({0: 'Not Survived', 1: 'Survived'})
    pred_df['Actual'] = pred_df['Actual'].map({0: 'Not Survived', 1: 'Survived'})

    st.dataframe(pred_df.head(20))


