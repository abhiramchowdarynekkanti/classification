import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

# ---- Streamlit UI ----
st.set_page_config(page_title="🧠 ML Classifier Dashboard", layout="centered")
st.title("🧠 Classification on Custom Dataset")
st.write("Upload a CSV dataset or test the Titanic dataset. This app trains multiple classifiers and shows results.")

# ---- Load Data ----
@st.cache_data
def load_sample_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)



st.subheader("📂 Upload Dataset or Use Sample")

use_sample = st.checkbox("Use Titanic Sample Dataset")
if use_sample:




    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")


else:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if not uploaded_file:
        st.stop()
    df = load_data(uploaded_file)

st.subheader("🔍 Raw Data Preview")
st.dataframe(df.head())

# ---- Select Target Column ----
target_column = st.selectbox("🎯 Select Target Column", df.columns)

# ---- Preprocessing ----
def preprocess_data(df, target_column):
    df = df.copy()
    y = df[target_column]
    X = df.drop(columns=[target_column])

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include='object').columns

    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col])

    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    return X, y

X, y = preprocess_data(df, target_column)

# ---- Train-test Split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Model Definitions ----
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

# ---- Training Mode ----
mode = st.radio("⚙️ Select Training Mode", ["Without Tuning", "With Tuning"])
results = []

with st.spinner("Training models..."):
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
                'Precision': round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'Recall': round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'F1 Score': round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'R² Score': round(r2, 4) if r2 is not None else '—'
            })
    else:
        for name, (model, params) in model_params.items():
            grid = GridSearchCV(model, params, scoring='f1_weighted', cv=3, n_jobs=-1)
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
                'Precision': round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'Recall': round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'F1 Score': round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                'R² Score': round(r2, 4) if r2 is not None else '—'
            })

# ---- Results Table ----
st.subheader("📈 Model Performance")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
st.dataframe(results_df)

best_model_name = results_df.iloc[0]['Model']
st.success(f"🏆 Best Model: **{best_model_name}** ({mode})")

# ---- Predictions Preview ----
st.subheader("🔮 Sample Predictions")

if mode == "Without Tuning":
    model = base_models[best_model_name]
    model.fit(X_train, y_train)
else:
    model, params = model_params[best_model_name]
    grid = GridSearchCV(model, params, scoring='f1_weighted', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

y_pred = model.predict(X_test)

preview = pd.DataFrame(X_test).copy()
preview['Actual'] = y_test.values
preview['Predicted'] = y_pred

st.dataframe(preview.head(20))
