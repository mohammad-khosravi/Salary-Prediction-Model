import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("data_jobs.csv")
df = df[df['salary_year_avg'].notnull()].copy()

# Preprocess job_skills
df['job_skills'] = df['job_skills'].fillna('')
df['job_skills'] = df['job_skills'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

# Features and target
features = ['job_title_short', 'job_location', 'job_schedule_type', 'job_work_from_home',
            'job_no_degree_mention', 'job_health_insurance', 'job_country', 'job_skills']
target = 'salary_year_avg'

# Sample and split
df_sample = df.sample(n=10000, random_state=42)
X = df_sample[features]
y = df_sample[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('title', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_title_short']),
    ('location', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_location']),
    ('schedule', OneHotEncoder(handle_unknown='ignore'), ['job_schedule_type']),
    ('country', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_country']),
    ('skills', CountVectorizer(max_features=100), 'job_skills'),
], remainder='passthrough')

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R²: {r2:.4f}")

# Plot predictions
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.3, color='teal')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Random Forest: Actual vs Predicted Salary")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.grid(True)
plt.tight_layout()
plt.show()