Website Link  https://classification-3spznfj4jz6b9rwugpl69y.streamlit.app/
🚢 Classification Dashboard
A complete machine learning pipeline to predict Titanic passenger survival. Compares models with and without hyperparameter tuning, presented through an interactive Streamlit dashboard.

📁 Project Structure
📊 Data Preparation: Cleans data, handles missing values, feature engineering (family_size), and encoding.

🤖 Modeling: Trains models with and without tuning using GridSearchCV.

📈 Evaluation: Displays performance metrics with Streamlit interface.

🧠 Algorithms Implemented
🔹 Logistic Regression
Type: Linear binary classifier

Tuned: C, penalty

🌲 Decision Tree
Type: Rule-based tree

Tuned: max_depth, min_samples_split

🌳 Random Forest
Type: Ensemble of trees

Tuned: n_estimators, max_depth, min_samples_split

💠 SVM (Support Vector Machine)
Type: Margin-based classifier

Tuned: C, kernel

⚡ Gradient Boosting
Type: Boosted weak learners

Tuned: n_estimators, learning_rate, max_depth

🛠️ Optimization Strategies
🔸 Without Tuning
Uses default model parameters

Fast and useful for baselines

🔹 With Tuning (GridSearchCV)
3-fold cross-validation

Optimizes for F1 Score

Reduced parameter grid for speed

📏 Evaluation Metrics
✅ Accuracy – Overall correctness

🎯 Precision – True positives out of predicted positives

📢 Recall – True positives out of actual positives

⚖️ F1 Score – Balance of precision and recall

📉 R² Score – Regression metric (comparison only)
