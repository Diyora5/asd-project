import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# --- CONFIG ---
DATA_PATH = 'data/autism_screening_dataset.csv'
MODELS_PATH = 'models/'

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH)

# --- 1. INTERACTIVE CLASS DISTRIBUTION ---
fig_class = px.histogram(df, x='Autism_Diagnosis', color='Autism_Diagnosis',
                        title='Class Distribution of ASD Traits',
                        labels={'Autism_Diagnosis': 'ASD Diagnosis'})
fig_class.write_html('interactive_class_distribution.html')

# --- 2. INTERACTIVE FEATURE CORRELATION HEATMAP ---
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_features].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu',
                    title='Feature Correlation Heatmap')
fig_corr.write_html('interactive_feature_correlation.html')

# --- 3. INTERACTIVE DEMOGRAPHIC DISTRIBUTIONS ---
if 'Age' in df.columns:
    fig_age = px.histogram(df, x='Age', nbins=30, title='Age Distribution')
    fig_age.write_html('interactive_age_distribution.html')
if 'Gender' in df.columns:
    fig_gender = px.histogram(df, x='Gender', color='Autism_Diagnosis', barmode='group',
                             title='Gender Distribution by Diagnosis')
    fig_gender.write_html('interactive_gender_distribution.html')

# --- 4. LOAD MODELS ---
model_files = glob.glob(os.path.join(MODELS_PATH, '*.pkl'))
models = {}
for mf in model_files:
    try:
        model_name = os.path.basename(mf).replace('.pkl', '')
        models[model_name] = joblib.load(mf)
    except Exception as e:
        print(f'Could not load {mf}: {e}')

# --- 5. PREPARE DATA FOR MODEL EVALUATION ---
X = df.drop(columns=['Autism_Diagnosis'])
y = df['Autism_Diagnosis']
if y.dtype == object:
    y = LabelBinarizer().fit_transform(y).ravel()

# --- 6. FEATURE IMPORTANCE (TREE MODELS) ---
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        fig_imp = px.bar(x=features, y=importances, title=f'Feature Importance: {name}',
                        labels={'x': 'Feature', 'y': 'Importance'})
        fig_imp.write_html(f'interactive_feature_importance_{name}.html')

# --- 7. ROC & AUC CURVES FOR ALL MODELS ---
roc_fig = go.Figure()
auc_scores = {}
acc_scores = {}
f1_scores = {}
for name, model in models.items():
    try:
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X)
        else:
            y_score = model.predict(X)
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = roc_auc
        acc_scores[name] = accuracy_score(y, model.predict(X))
        f1_scores[name] = f1_score(y, model.predict(X))
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.3f})'))
    except Exception as e:
        print(f'Could not compute ROC for {name}: {e}')
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
roc_fig.update_layout(title='ROC Curves for All Models', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.write_html('interactive_roc_curves.html')

# --- 8. MODEL COMPARISON BARPLOTS ---
comp_df = pd.DataFrame({
    'Model': list(auc_scores.keys()),
    'AUC': list(auc_scores.values()),
    'Accuracy': [acc_scores[m] for m in auc_scores.keys()],
    'F1': [f1_scores[m] for m in auc_scores.keys()]
})
for metric in ['AUC', 'Accuracy', 'F1']:
    fig = px.bar(comp_df, x='Model', y=metric, title=f'Model Comparison: {metric}')
    fig.write_html(f'interactive_model_comparison_{metric.lower()}.html')

print('All interactive plots saved as HTML files.') 