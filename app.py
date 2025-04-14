import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.feature_selection import VarianceThreshold

st.set_page_config(page_title="\U0001F9E0 ML Classifier Dashboard", layout="centered")
st.title("\U0001F9E0 Classification on Custom Dataset")
st.write("Upload a CSV dataset or test the Titanic dataset. This app trains multiple classifiers and shows results.")

@st.cache_data
def load_sample_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

def load_uploaded_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read uploaded CSV: {e}")
        st.stop()

st.subheader("\U0001F4C2 Upload Dataset or Use Sample")

use_sample = st.checkbox("Use Titanic Sample Dataset")

if use_sample:
    df = load_sample_data()
    target_column = st.selectbox("\U0001F3AF Select Target Column", df.columns, index=df.columns.get_loc('Survived'))
else:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = load_uploaded_data(uploaded_file)
        target_column = st.selectbox("\U0001F3AF Select Target Column", df.columns)
    else:
        st.error("Please upload a file or select the sample dataset.")
        st.stop()

st.subheader("\U0001F50D Raw Data Preview")
st.dataframe(df.head())

def preprocess_data(df, target_column):
    df = df.copy()
    y = df[target_column]
    X = df.drop(columns=[target_column])

    if y.nunique() < 2:
        st.error("❌ Target column must have at least two unique values.")
        st.stop()

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include='object').columns

    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col])

    if X.shape[1] == 0:
        st.error("❌ No usable features after preprocessing. Check your dataset.")
        st.stop()

    X = StandardScaler().fit_transform(X)
    X = VarianceThreshold().fit_transform(X)
    return pd.DataFrame(X), y

X, y = preprocess_data(df, target_column)

if y.nunique() < 2:
    st.error("❌ Classification requires at least two classes in the target column.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier()
}

model_params = {
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

mode = st.radio("⚙️ Select Training Mode", ["Without Tuning", "With Tuning (Please Wait , takes a bit Longer)"])
results = []

with st.spinner("Training models..."):
    if mode == "Without Tuning":
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            try:
                r2 = r2_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
            except:
                r2 = None
            results.append({
                'Model': name,
                'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                'Precision': round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'Recall': round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'F1 Score': round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'R² Score': round(r2, 4) if r2 is not None else '—'
            })
    else:
        for name, (model, params) in model_params.items():
            try:
                grid = GridSearchCV(model, params, scoring='f1_weighted', cv=StratifiedKFold(n_splits=3), n_jobs=-1, error_score=0)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
                try:
                    r2 = r2_score(y_test, best_model.predict_proba(X_test)[:, 1]) if hasattr(best_model, "predict_proba") else None
                except:
                    r2 = None
                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'Precision': round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    'Recall': round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    'F1 Score': round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    'R² Score': round(r2, 4) if r2 is not None else '—'
                })
            except ValueError as e:
                st.warning(f"⚠️ Skipping {name} due to error: {e}")

st.subheader("\U0001F4C8 Model Performance")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
st.dataframe(results_df)

if not results_df.empty:
    best_model_name = results_df.iloc[0]['Model']
    st.success(f"\U0001F3C6 Best Model: **{best_model_name}** ({mode})")

    st.subheader("\U0001F52E Sample Predictions")

    if mode == "Without Tuning":
        model = base_models[best_model_name]
        model.fit(X_train, y_train)
    else:
        model, params = model_params[best_model_name]
        grid = GridSearchCV(model, params, scoring='f1_weighted', cv=3, n_jobs=-1, error_score=0)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

    y_pred = model.predict(X_test)
    preview = pd.DataFrame(X_test).copy()
    preview['Actual'] = y_test.values
    preview['Predicted'] = y_pred

    st.dataframe(preview.head(20))
else:
    st.error("❌ No valid models could be trained.")
