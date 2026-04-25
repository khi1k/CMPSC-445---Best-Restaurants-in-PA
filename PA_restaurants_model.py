import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

print("PA Restaurant rating predictions - Model comparisons (Regression, XGBoost, Random Forest)")

#Loads the preprocessed dataset
df = pd.read_csv('restaurants_pa_model_ready.csv')
print(f"Original shape: {df.shape}")

df = df.dropna(subset=['rating'])
print(f"After dropping missing ratings: {df.shape}")

#Sets the target value, as well as columns to exclude
target_col = 'rating'
exclude_cols = [target_col, 'Unnamed: 0', 'beam_id', 'title', 'source_file']

cat_cols = [col for col in df.columns if col.startswith('cat_') and col not in exclude_cols]
other_cols = [col for col in df.columns if col not in exclude_cols + cat_cols + [target_col]]
print(f"Total category columns: {len(cat_cols)}")
print(f"Non-category columns: {len(other_cols)}")

#Feature selection: use Random Forest to pick top 15 non‑categorical features
X_all = df[other_cols]
y = df[target_col]
imputer_temp = SimpleImputer(strategy='mean')
X_all_imputed = imputer_temp.fit_transform(X_all)

rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_temp.fit(X_all_imputed, y)
importances = rf_temp.feature_importances_
top_k = 15
indices = np.argsort(importances)[::-1][:top_k]
selected_other_cols = [other_cols[i] for i in indices]
final_features = cat_cols + selected_other_cols
print(f"Total features used: {len(final_features)}")

X = df[final_features]
y = df[target_col]
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

#Training/testing data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Shared imputer
imputer = SimpleImputer(strategy='mean')

#Linear Regression pipeline
lr_pipeline = Pipeline([
    ('imputer', imputer),
    ('model', LinearRegression())
])

#Random Forest pipeline
rf_pipeline = Pipeline([
    ('imputer', imputer),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    ))
])

#XGBoost pipeline with hyperparameter tuning
xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_pipeline = Pipeline([
    ('imputer', imputer),
    ('model', xgb_base)
])
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0]
}
print("Tuning XGBoost hyperparameters")
xgb_grid = GridSearchCV(xgb_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"Best XGBoost params: {xgb_grid.best_params_}")
print(f"Best CV R²: {xgb_grid.best_score_:.4f}")

#Train and evaluate all models
models = {
    'Linear Regression': lr_pipeline,
    'Random Forest': rf_pipeline,
    'XGBoost': best_xgb
}
results = []
best_model_name = None
best_test_r2 = -np.inf

print("Model Evaluation on Testing Set")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results.append({
        'Model': name,
        'Test R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV Mean R²': cv_scores.mean(),
        'CV Std R²': cv_scores.std()
    })
    print(f"\n{name}")
    print(f"   Test R²:        {r2:.4f}")
    print(f"   RMSE:           {rmse:.3f} stars")
    print(f"   MAE:            {mae:.3f} stars")
    print(f"   CV R² (train):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if r2 > best_test_r2:
        best_test_r2 = r2
        best_model_name = name

#Comparison table
print("\nModel Comparisons - ")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\nBest model based on test R²: {best_model_name} (R² = {best_test_r2:.4f})")

#Plot visualizations for the best model
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, y_pred_best, alpha=0.6, edgecolors='k', c='teal')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Rating')
ax.set_ylabel('Predicted Rating')
ax.set_title(
    f'{best_model_name} (Best Model)\nR² = {best_test_r2:.3f}, MAE = {results_df[results_df.Model == best_model_name].MAE.values[0]:.2f} stars')
plt.tight_layout()
plt.savefig('actual_vs_predicted_best.png', dpi=150)
plt.show()
print("Saved actual_vs_predicted_best.png")

residuals = y_test - y_pred_best
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(residuals, bins=20, kde=True, color='purple', edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--')
ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.set_title(f'Residual Distribution – {best_model_name}')
plt.savefig('residuals_best.png', dpi=150)
plt.show()
print("Saved residuals_best.png")

#Feature importance for RF and XGB
if best_model_name in ['Random Forest', 'XGBoost']:
    importances_best = best_model.named_steps['model'].feature_importances_
    indices_best = np.argsort(importances_best)[::-1][:15]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(indices_best)), importances_best[indices_best], color='steelblue')
    ax.set_yticks(range(len(indices_best)))
    ax.set_yticklabels([final_features[i] for i in indices_best])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 15 Features – {best_model_name}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_best.png', dpi=150)
    plt.show()
    print("Saved feature_importance_best.png")

#Save the best model for deployment
joblib.dump(best_model, 'best_restaurant_pipeline.pkl')
joblib.dump(final_features, 'model_features.pkl')
print(f"Best model ({best_model_name}) saved as 'best_restaurant_pipeline.pkl'")
print("Feature list saved as 'model_features.pkl'")

#Generate predictions for all restaurants using the best model
df_raw = pd.read_csv('restaurants_pa_clean.csv')
missing_features = [f for f in final_features if f not in df_raw.columns]
if missing_features:
    print(f"Warning: Missing columns: {missing_features}")
else:
    X_all_rest = df_raw[final_features].copy()
    y_pred_all = best_model.predict(X_all_rest)
    df_raw['predicted_rating'] = y_pred_all
    df_raw.to_csv('restaurants_with_predictions.csv', index=False)
    print(f"Saved 'restaurants_with_predictions.csv' with predictions for {len(y_pred_all)} restaurants.")
