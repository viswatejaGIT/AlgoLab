# importing Ui,ML requirements
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# importing Ml libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold

# importing Ml models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier


# page setup
st.set_page_config(page_title='Algolab', layout='wide')

# css
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2E96AB;
    margin-bottom: 1rem;
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

# defining model
Clasification_models = {
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()  
}

Regression_models = {
    'Linear regression' : LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Support Vector Regression': SVR(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    'Ridge Regression': Ridge(random_state=42)
}


# helper functions - upload file, do staticits, preprocessing, modeling part

#data loading function
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# data preprocessing function
def data_preprocessing(df, target_col, feature_cols):
    
    #get the data
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # fill missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            most_common = X[col].mode()
            if len(most_common) > 0:
                fill_with = most_common[0]
            else:
                fill_with = 'unknown'
            X[col] = X[col].fillna(fill_with)
        else:
            X[col] = X[col].fillna(X[col].mean())
    
    # handling catogorical value
    catagorical_cols = X.select_dtypes(include=['object']).columns
    for col in catagorical_cols:
        unique = X[col].nunique()
        
        if unique <= 10:
            # if unique values in col are <10, use one-hot
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            # we use frequency encoding
            freq = X[col].value_counts().to_dict()
            X[col] = X[col].map(freq)
    
    #ensure numeric values are actual numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except ValueError:
                pass
        
    # handling outliers (IQR method)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        X[col] = np.clip(X[col], lower_bound, upper_bound)
    
    
    return X,y

# Traning data and evaluating models

def train_and_evaluate_models(X, y, problem_type):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    #choose model based on problem type
    if problem_type == "Classification":
        models = Clasification_models
    else:
        models = Regression_models
    
    results = []
    predictions = {}

    # Train each model
    for name, model in models.items():
        # scale data for certain models
        if name in ['Logistic Regression', 'Support Vector Machine', 'Support Vector Regression', 'K-Nearest Neighbors']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # store predictions    
        predictions[name] = y_pred

        # Calculate metrics
        if problem_type == 'Classification':
            accuracy = accuracy_score(y_test, y_pred)
            Precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'Model': name,
                'Accuracy': round(accuracy, 4),
                'Precision': round(Precision, 4),
                'Recall': round(recall, 4),
                'F1-Score': round(f1, 4)
            })
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse) 
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Model' : name,
                'RMSE' : round(rmse, 4),
                'MAE' : round(mae, 4),
                'R2 Score' : round(r2, 4)
            })
    
    return pd.DataFrame(results), predictions, y_test

# ranking the best model 
def calculate_composite_score(results, problem_type):
    if problem_type == 'Classification':
        # Normalize metrics to 0-1 scale
        results['Accuracy_norm'] = results['Accuracy'] / results['Accuracy'].max()
        results['Precision_norm'] = results['Precision'] / results['Precision'].max()
        results['Recall_norm'] = results['Recall'] / results['Recall'].max()
        results['F1_norm'] = results['F1-Score'] / results['F1-Score'].max()
        
        # Weighted composite score (F1 gets highest weight)
        results['Composite_Score'] = (
            0.2 * results['Accuracy_norm'] + 
            0.25 * results['Precision_norm'] + 
            0.25 * results['Recall_norm'] + 
            0.3 * results['F1_norm']
        )
    else:
        # For regression - normalize R2 (higher is better), invert RMSE/MAE (lower is better)
        results['R2_norm'] = results['R2 Score'] / results['R2 Score'].max()
        results['RMSE_norm'] = results['RMSE'].min() / results['RMSE']  # Inverted
        results['MAE_norm'] = results['MAE'].min() / results['MAE']    # Inverted
        
        # Weighted composite score
        results['Composite_Score'] = (
            0.5 * results['R2_norm'] + 
            0.25 * results['RMSE_norm'] + 
            0.25 * results['MAE_norm']
        )
    
    return results.sort_values('Composite_Score', ascending=False)


 # ------------------------------- Main App look ----------------------------------

st.markdown('<h1 class="main-header">AlgoLab</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Machine Learning Model Comparison Platform</p>', unsafe_allow_html=True)

#sidebars
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
    st.text("* Logistic Regression\n* Random Forest Classifier\n* Support Vector Classifier\n* Gradient Boosting Classifier\n* K-Nearest Neighbors Classifier")
    
    st.header("Regression Models")
    st.text("* Linear Regression\n* Random Forest Regressor\n* Support Vector Regressor\n* Gradient Boosting Regressor\n* Ridge Regressor")

# ----------------------------- Tab1 [Data Upload and View] -------------------------------------------------

t1, t2 = st.tabs(["Data Upload and View", "Model Analysis"])

with t1:
    st.subheader("upload Your Dataset")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        # load data
        df = load_data(uploaded_file)
        st.success(f"Data loaded, its Shape: {df.shape}")

        # data first 5 rows
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Missing Values", df.isnull().sum().sum())
        with col2:
            st.metric("Total Duplicate Rows", df.duplicated().sum())
        with col3:
            st.metric("Total Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Total Categorical Columns", len(df.select_dtypes(include=['object']).columns))

        #data info
        st.subheader("Dataset Information")
        st.write("**Data Types:**")
        info_df = pd.DataFrame({
            'column': df.columns,
            'Type': df.dtypes,
            'Missing':df.isnull().sum()
        })
        st.dataframe(info_df, use_container_width=True)

        st.write("**Statistics:**")
        st.dataframe(df.describe(), use_container_width=True)

        # Memory Usage
        st.write("**Memory Usage:**")
        st.text(f"Dataset size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Unique Values per Column
        st.write("**Unique Values:**")
        unique_df = pd.DataFrame({
            'Column': df.columns,
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Unique percentage(%)': [round(df[col].nunique()/len(df)*100, 2) for col in df.columns]
        })
        st.dataframe(unique_df, use_container_width=True)

        #model setup
        st.subheader("Model Configuration")

        col1,col2 = st.columns(2)

        with col1:
            target_col = st.selectbox("Select Target Variable", df.columns)
        
        with col2:
            problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])    
        
        #feature selection
        st.subheader("Select Features")
        availabe = [col for col in df.columns if col != target_col]
        selected = st.multiselect("Select Features", availabe, default=availabe[:15])

        #Save to secssion state
        if selected:
            st.session_state['df'] = df
            st.session_state['target'] = target_col
            st.session_state['features'] = selected
            st.session_state['problem_type'] = problem_type
            st.success(f"Configuration saved!  || Target: {target_col} || Features: {len(selected)} || problem_type: {problem_type}")
            
            # Navigation note
            st.markdown("---")
            st.markdown(
                '<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">'
                '<h4 style="margin: 0; color: white;">Ready for Analysis!</h4>'
                '<p style="margin: 0.5rem 0 0 0; color: white;">Please navigate to the <strong>"Model Analysis"</strong> tab to perform ML model comparison and get insights.</p>'
                '</div>', 
                unsafe_allow_html=True
            )


# ------------------------- tab2 [Model Analysis] -----------------------------------------

with t2:
    st.subheader("Model Comparision Results")

    if 'df' not in st.session_state:
        st.warning("Please upload and configure your data first")
    else:
        df = st.session_state['df']
        target = st.session_state['target']
        features = st.session_state['features']
        problem_type = st.session_state['problem_type']

        st.info(f"Target: {target} | Features: {len(features)} | Type: {problem_type}")
    
    if st.button("Start Analysis", type="primary"):
        with st.spinner("Training model..."):
            
            # data preprocessing
            X,y = data_preprocessing(df, target, features)

            # model training and evaluation
            model_results = train_and_evaluate_models(X,y, problem_type)

            # Save results
            st.session_state['results'] = model_results[0]
            st.session_state['predictions'] = model_results[1] 
            st.session_state['y_test'] = model_results[2]  
        
    #show results
    if 'results' in st.session_state:
        results = st.session_state['results']
        predictions = st.session_state['predictions']
        y_test = st.session_state['y_test']
        
        # Apply composite scoring FIRST
        results = calculate_composite_score(results, problem_type)
        
        st.subheader("Performance Results")
        st.dataframe(results, use_container_width=True)

        # Visualizations
        st.subheader("Performance Charts")

        if problem_type == 'Classification':
            # Classification metrics chart
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'))

            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            positions = [(1,1), (1,2), (2,1), (2,2)]

            for metric, pos in zip(metrics, positions):
                fig.add_trace(
                    go.Bar(x=results['Model'], y=results[metric], name=metric, showlegend=False),
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
                    go.Bar(x=results['Model'], y=results[metric], name=metric, showlegend=False),
                    row=1, col=i+1
                )

            fig.update_layout(height=400, title_text="Regression Metrics")
            st.plotly_chart(fig, use_container_width=True)
        
        # Top models
        st.subheader('Top Models')
        st.info("**Composite Score Explanation**: This is a single number (0 to 1) that combines all individual performance metrics into one overall ranking. Instead of looking at different evaluation scores separately, the composite score gives you one final score to easily compare which model performs best overall. A score closer to 1.0 means better performance.")
        top_2 = results.head(2)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="best-model">', unsafe_allow_html=True)
            st.write(f"**1st Place: {top_2.iloc[0]['Model']}**")
            st.write(f"Composite Score: {top_2.iloc[0]['Composite_Score']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="best-model">', unsafe_allow_html=True)
            st.write(f"**2nd Place: {top_2.iloc[1]['Model']}**")
            st.write(f"Composite Score: {top_2.iloc[1]['Composite_Score']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("--------------------------------------------------------------------------------")
st.markdown("**AlgoLab** - Machine Learning Model Comparison Platform", unsafe_allow_html=True)