# AlgoLab - ML Model Comparison Tool
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix

# Import ML models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier

# Page setup
st.set_page_config(page_title="AlgoLab", layout="wide")

# CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2E86AB;
    margin-bottom: 2rem;
}
.best-model {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Define models
classification_models = {
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

regression_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Support Vector Regression': SVR(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    'Ridge Regression': Ridge(random_state=42)
}

# Functions
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

## Clean and prepare data for ML
def prepare_data(df, target_col, feature_cols):
    # Get features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Fill missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        else:
            X[col] = X[col].fillna(X[col].mean())
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y

class ModelResults:
    def __init__(self, results_df, predictions, y_test):
        self.results_df = results_df
        self.predictions = predictions
        self.y_test = y_test

## Train all models and return results

def train_models(X, y, problem_type):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose models based on problem type
    if problem_type == "Classification":
        models = classification_models
    else:
        models = regression_models
    
    results = []
    predictions = {}
    
    # Train each model
    for name, model in models.items():
        # Scale data for certain models
        if name in ['Logistic Regression', 'Support Vector Machine', 'Support Vector Regression', 'K-Nearest Neighbors']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Store predictions
        predictions[name] = y_pred
        
        # Calculate metrics
        if problem_type == "Classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'Model': name,
                'Accuracy': round(accuracy, 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1-Score': round(f1, 4)
            })
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R2 Score': round(r2, 4)
            })
    
    return ModelResults(pd.DataFrame(results), predictions, y_test)

# Main App
st.markdown('<h1 class="main-header">AlgoLab</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Machine Learning Model Comparison Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Features")
    st.info("""
    - CSV/Excel Data Upload
    - Dataset Statistical Info
    - 5 Top ML Algorithm Comparison
    - Performance Visualizations
    - Best Model Recommendations
    """)
    
    st.header("Classification Models")
    st.text("• Logistic Regression\n• Random Forest Classifier\n• Support Vector Classifier\n• Gradient Boosting Classifier\n• K-Nearest Neighbors Classifier")
    
    st.header("Regression Models")
    st.text("• Linear Regression\n• Random Forest Regressor\n• Support Vector Regressor\n• Gradient Boosting Regressor\n• Ridge Regressor")

# Main tabs
tab1, tab2 = st.tabs(["Data Upload", "Model Analysis"])

with tab1:
    st.subheader("Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        st.success(f"Data loaded! Shape: {df.shape}")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Basic stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        
        # Data info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Missing': df.isnull().sum()
            })
            st.dataframe(info_df)
        
        with col2:
            st.write("**Statistics:**")
            st.dataframe(df.describe())
        
        # Model setup
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("Select Target Variable", df.columns)
            
            if target_col:
                unique_vals = df[target_col].nunique()
                st.write(f"**Target Info:** {unique_vals} unique values")
                
                if unique_vals <= 10:
                    sample_vals = df[target_col].unique()[:5]
                    st.write(f"**Sample values:** {list(sample_vals)}")
        
        with col2:
            problem_type = st.radio("Problem Type", ["Classification", "Regression"])
        
        # Feature selection
        st.subheader("Select Features")
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect("Choose features for training", available_features, default=available_features[:10])
        
        # Save to session state
        if selected_features:
            st.session_state['df'] = df
            st.session_state['target'] = target_col
            st.session_state['features'] = selected_features
            st.session_state['problem_type'] = problem_type
            st.success(f"Configuration saved! Target: {target_col}, Features: {len(selected_features)}")

with tab2:
    st.subheader("Model Comparison Results")
    
    if 'df' not in st.session_state:
        st.warning("Please upload and configure your data first.")
    else:
        df = st.session_state['df']
        target_col = st.session_state['target']
        features = st.session_state['features']
        problem_type = st.session_state['problem_type']
        
        st.info(f"Target: {target_col} | Features: {len(features)} | Type: {problem_type}")
        
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Training models..."):
                # Prepare data
                X, y = prepare_data(df, target_col, features)
                
                # Train models
                model_results = train_models(X, y, problem_type)
                
                # Save results
                st.session_state['results'] = model_results.results_df
                st.session_state['predictions'] = model_results.predictions
                st.session_state['y_test'] = model_results.y_test
        
        # Show results
        if 'results' in st.session_state:
            results_df = st.session_state['results']
            predictions = st.session_state['predictions']
            y_test = st.session_state['y_test']
            
            # Sort results
            if problem_type == "Classification":
                results_df = results_df.sort_values('Accuracy', ascending=False)
            else:
                results_df = results_df.sort_values('R2 Score', ascending=False)
            
            st.subheader("Performance Results")
            st.dataframe(results_df)
            
            # Visualizations
            st.subheader("Performance Charts")
            
            if problem_type == "Classification":
                # Classification metrics chart
                fig = make_subplots(rows=2, cols=2, subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'))
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                positions = [(1,1), (1,2), (2,1), (2,2)]
                
                for metric, pos in zip(metrics, positions):
                    fig.add_trace(
                        go.Bar(x=results_df['Model'], y=results_df[metric], name=metric, showlegend=False),
                        row=pos[0], col=pos[1]
                    )
                
                fig.update_layout(height=600, title_text="Classification Metrics")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Regression metrics chart
                fig = make_subplots(rows=1, cols=3, subplot_titles=('R2 Score', 'RMSE', 'MAE'))
                
                metrics = ['R2 Score', 'RMSE', 'MAE']
                
                for i, metric in enumerate(metrics):
                    fig.add_trace(
                        go.Bar(x=results_df['Model'], y=results_df[metric], name=metric, showlegend=False),
                        row=1, col=i+1
                    )
                
                fig.update_layout(height=400, title_text="Regression Metrics")
                st.plotly_chart(fig, use_container_width=True)
            
            # Best models
            st.subheader("Top Models")
            
            top_2 = results_df.head(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="best-model">', unsafe_allow_html=True)
                st.write(f"**1st Place: {top_2.iloc[0]['Model']}**")
                if problem_type == "Classification":
                    st.write(f"Accuracy: {top_2.iloc[0]['Accuracy']}")
                else:
                    st.write(f"R2 Score: {top_2.iloc[0]['R2 Score']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="best-model">', unsafe_allow_html=True)
                st.write(f"**2nd Place: {top_2.iloc[1]['Model']}**")
                if problem_type == "Classification":
                    st.write(f"Accuracy: {top_2.iloc[1]['Accuracy']}")
                else:
                    st.write(f"R2 Score: {top_2.iloc[1]['R2 Score']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional visualizations
            if problem_type == "Regression":
                st.subheader("Prediction vs Actual - Top Models")
                
                top_models = results_df.head(2)['Model'].tolist()
                cols = st.columns(2)
                
                for i, model_name in enumerate(top_models):
                    with cols[i]:
                        st.write(f"**{model_name}**")
                        
                        y_pred = predictions[model_name]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
                        
                        # Perfect line
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                               mode='lines', name='Perfect Line', line=dict(color='red', dash='dash')))
                        
                        fig.update_layout(height=300, xaxis_title="Actual", yaxis_title="Predicted", showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            
            elif problem_type == "Classification":
                st.subheader("Confusion Matrices - Top Models")
                
                top_models = results_df.head(2)['Model'].tolist()
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=top_models)
                
                for i, model_name in enumerate(top_models):
                    y_pred = predictions[model_name]
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig.add_trace(
                        go.Heatmap(z=cm, text=cm, texttemplate="%{text}", colorscale='Blues', showscale=i==0),
                        row=1, col=i+1
                    )
                
                fig.update_layout(height=300, title_text="Confusion Matrices")
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**AlgoLab** - Machine Learning Model Comparison Platform", unsafe_allow_html=True)