# Salary-Prediction-Model
Salary Prediction Model - Linear Regression vs. Random Forest vs. XGBoost
-
= Goal: Predict job salary based on job title, location, skills, and employment type.

= Model: Linear Regression, Random Forest, XGBoost

= Skills: Data Cleaning, Feature Engineering, Machine Learning, Model Deployment

This analysis is based on the dataset provided by Luke Barousse, available on Hugging Face: Data
Jobs Dataset.
# Salary Prediction Model — Linear Regression vs. Random Forest vs. XGBoost

Goal:: Predict job salary based on job title, location, skills, and employment type.
Model: Linear Regression, Random Forest, XGBoost
Skills: Data Cleaning, Feature Engineering, Machine Learning, Model Deployment

This analysis is based on the dataset provided by Luke Barousse, available on Hugging Face: [Data Jobs Dataset](https://huggingface.co/datasets/lukebarousse/data_jobs).

---

First things first, let’s take a quick look at the correlation matrix to have a rough idea about which features to use in the model:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess
df = pd.read_csv("data_jobs.csv")
df = df[df['salary_year_avg'].notnull()].copy()
df['job_skills'] = df['job_skills'].fillna('')
df['job_skills'] = df['job_skills'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

# Sample
df_sample = df.sample(n=10000, random_state=42)

# Numeric conversion
df_sample['skills'] = df_sample['job_skills'].apply(lambda x: len(x.split()))
df_sample['title'] = df_sample['job_title_short'].astype('category').cat.codes
df_sample['location'] = df_sample['job_location'].astype('category').cat.codes
df_sample['schedule'] = df_sample['job_schedule_type'].astype('category').cat.codes
df_sample['country'] = df_sample['job_country'].astype('category').cat.codes

# Rename and select relevant columns
df_corr = df_sample.rename(columns={
    'salary_year_avg': 'salary',
    'job_work_from_home': 'remote',
    'job_no_degree_mention': 'no_degree',
    'job_health_insurance': 'insurance',
})[['salary', 'title', 'location', 'schedule', 'remote', 'no_degree', 'insurance', 'country', 'skills']]

# Compute correlation matrix
corr_matrix = df_corr.corr()

# Plot full 2D heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Reds', vmin=0, vmax=1, fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
```

![Correlation Heatmap](https://github.com/user-attachments/assets/a1898813-1401-4403-a4fe-65fe5721bdee)

### Let’s Start with the Linear Regression

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv("data_jobs.csv")
df = df[df['salary_year_avg'].notnull()].copy()

# Clean and process job_skills
df['job_skills'] = df['job_skills'].fillna('')
df['job_skills'] = df['job_skills'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

# Feature selection
features = ['job_title_short', 'job_location', 'job_schedule_type', 'job_work_from_home',
            'job_no_degree_mention', 'job_health_insurance', 'job_country', 'job_skills']
target = 'salary_year_avg'

# Sample for efficiency
df_sample = df.sample(n=10000, random_state=42)
X = df_sample[features]
y = df_sample[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('title', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_title_short']),
    ('location', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_location']),
    ('schedule', OneHotEncoder(handle_unknown='ignore'), ['job_schedule_type']),
    ('country', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_country']),
    ('skills', CountVectorizer(max_features=100), 'job_skills'),
], remainder='passthrough')

# Pipeline with Linear Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression RMSE: {rmse:.2f}")
print(f"Linear Regression R²: {r2:.4f}")

# Plot predictions
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.3, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Linear Regression: Actual vs Predicted Salary")
plt.grid(True)
plt.tight_layout()
plt.show()
```

![Linear Regression- Actual vs Predicted Salary](https://github.com/user-attachments/assets/92003080-c0c2-4891-81e0-685a7c5798d5)

### A Few Observation

**Clustering around lower salary ranges**

- Most points are concentrated below $200,000.
- This indicates most of the dataset contains jobs in the low-to-mid salary range.

**Model follows the trend but underperforms at higher salaries**

- The spread increases as salary increases.
- The model **under predicts many high-salary jobs** (dots are below the diagonal).
- This is typical for Linear Regression, which struggles with **outliers and non-linear relationships**.

**Prediction accuracy is decent but not perfect**

- You can see a general diagonal trend (which is good), but with quite a bit of scatter.
- The further a dot is from the diagonal line, the less accurate that prediction was.

### Why It Happens

- **Linear Regression assumes a linear relationship** — salary is likely influenced by complex, nonlinear factors (e.g., rare skills or job titles).
- **High salaries are rare** — the model had fewer examples to learn from in that range.
- **Some features like `skills` are reduced to word counts**, which simplifies rich information.

### Next Step — Random Forest or Gradient Boosted Trees

- Let’s try **Random Forest** or **Gradient Boosted Trees** — they handle non-linear patterns and outliers much better.

```python
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
```

![Random Forest- Actual vs Predicted Salary](https://github.com/user-attachments/assets/a0a07384-5efa-4f4f-9a83-636604973ed5)

### Some Observations

**Tighter cluster along the diagonal**

- Compared to Linear Regression, **more points are closer to the ideal line**.
- Suggests Random Forest **makes better predictions overall**, especially in common salary ranges.

**Less spread at higher salaries**

- Linear Regression showed more wild predictions for high-salary jobs.
- Random Forest reduces this a bit — likely due to its ability to model **nonlinear relationships** and **interactions** between features.

**Still under predicts high-end salaries**

- Just like Linear Regression, it still **underestimates top salaries** (dots below the diagonal at the high end).
- Possible causes:
    - Fewer training examples in high-salary ranges.
    - Bias in the data or insufficient features describing high-paying roles (like rare senior roles or niche skills).

**Better generalization for mid-range**

- The model generalizes pretty well around $75k–$175k, where the data is denser.

### Quick Comparison to Linear Regression

| Feature | Linear Regression | Random Forest |
| --- | --- | --- |
| Handles Non-linearity | ❌ No | ✅ Yes |
| Sensitive to outliers | ✅ Yes | ⚠️ Somewhat |
| Higher salary prediction | ❌ Worse | ✅ Slightly better |
| Model interpretability | ✅ Easy | ❌ Harder |
| Accuracy on this task | ⚠️ Ok | ✅ Better |

### Now It’s Time for Gradient Boosted Trees

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor  # ✅ XGBoost

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

# XGBoost Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective='reg:squarederror', n_jobs=-1, random_state=42))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"XGBoost RMSE: {rmse:.2f}")
print(f"XGBoost R²: {r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.3, color='darkorange')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("XGBoost: Actual vs Predicted Salary")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.grid(True)
plt.tight_layout()
plt.show()
```

![XBoost- Actual vs Predicted Salary](https://github.com/user-attachments/assets/eb64ff5c-8edd-4685-9c54-2fd430b3b735)


### Some Observations

**Tight Cluster Bottom Left:**

- Most predictions and actual values are clustered around $50k–$200k.
- This suggests the majority of the dataset is within this range (normal for most job listings).

**Performance:**

- Points **close to the line** = good prediction
- Points **above the line** = model overestimated salary
- Points **below the line** = model underestimated salary

**Spread vs Other Models:**

- Compared to **Linear Regression**, XGBoost usually shows better performance in:
    - Capturing non-linearities
    - Reducing over/under prediction for outliers
- Compared to Random Forest, XGBoost tends to generalize better and avoid overfitting on large feature sets.

---

### Model Performance Comparison

| Model | RMSE (↓ better) | R² Score (↑ better) |
| --- | --- | --- |
| **XGBoost** | **43,230.12** | **0.2884** |
| Random Forest | 44,400.47 | 0.2493 |
| Linear Regression | 44,572.18 | 0.2435 |

### Interpretation:

- **XGBoost performs the best overall**:
    - It has the **lowest RMSE**, meaning its predictions are on average closest to the true salary values.
    - It also has the **highest R²**, meaning it explains the most variance in the data.
- **Random Forest** performs second best. It is close to XGBoost but lags behind a bit in both metrics.
- **Linear Regression** performs the worst of the three, which is expected for a model that assumes linear relationships and can't capture complex patterns.

### Takeaway:

XGBoost is the best choice here if you want to maximize prediction accuracy. It captures nonlinearities and interactions more effectively than both Random Forest and Linear Regression.

![f6d8f9ed-4982-4966-a5e1-4ad1d7f38d1a](https://github.com/user-attachments/assets/0536547e-8ccb-4266-808b-54e890e0358a)

### **Feature Importance (for XGBoost & Random Forest)**

```python
importances = model.named_steps['regressor'].feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color="teal")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
```
