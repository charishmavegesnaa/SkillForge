import streamlit as st
import pandas as pd
import pickle
import joblib

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('student_logreg_model.pkl', 'rb') as file:
        return pickle.load(file)

model_data = load_model()

st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📊 Student Information")
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.sidebar.slider("Age", 15, 22, 17)
    studytime = st.sidebar.slider("Study Time (1-4)", 1, 4, 2)
    failures = st.sidebar.slider("Past Failures (0-3)", 0, 3, 0)

with col2:
    absences = st.sidebar.slider("Absences (0-32)", 0, 32, 4)
    freetime = st.sidebar.slider("Free Time (1-5)", 1, 5, 3)
    goout = st.sidebar.slider("Going Out (1-5)", 1, 5, 2)

col1, col2 = st.sidebar.columns(2)
with col1:
    dalc = st.sidebar.slider("Workday Alcohol (1-5)", 1, 5, 1)
with col2:
    walc = st.sidebar.slider("Weekend Alcohol (1-5)", 1, 5, 2)

g1 = st.sidebar.slider("1st Period Grade (G1, 0-20)", 0, 20, 12)
g2 = st.sidebar.slider("2nd Period Grade (G2, 0-20)", 0, 20, 13)

# Predict button
if st.sidebar.button("🔮 Predict Learning Level", type="primary"):
    student_data = {
        'age': age, 'studytime': studytime, 'failures': failures, 
        'absences': absences, 'G1': g1, 'G2': g2, 
        'freetime': freetime, 'goout': goout, 'Dalc': dalc, 'Walc': walc
    }
    
    # Preprocess
    df_new = pd.DataFrame([student_data])
    X_new = df_new[model_data['features']]
    X_new_poly = model_data['poly_features'].transform(X_new)
    X_new_scaled = model_data['scaler'].transform(X_new_poly)
    
    # Predict
    prediction = model_data['model'].predict(X_new_scaled)[0]
    probability = model_data['model'].predict_proba(X_new_scaled)[0]
    
    level = model_data['target_mapping'][prediction]
    
    # Main results
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 **Prediction Result**")
        
        # Level card
        level_colors = {
            'Beginner': '🔴 #ff6b6b',
            'Intermediate': '🟡 #feca57', 
            'Advanced': '🟢 #51cf66'
        }
        
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; font-size: 2.5rem;">{level}</h2>
            <p style="margin: 0; font-size: 1.2rem;">Predicted Learning Level</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"**Model Accuracy:** {model_data['accuracy']:.1%}")
    
    # Probabilities
    st.markdown("### 📈 Confidence Scores")
    prob_df = pd.DataFrame({
        'Level': list(model_data['target_mapping'].values()),
        'Probability': probability
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_df.set_index('Level'))
    
    # Student data table
    st.markdown("### 👤 Student Profile")
    st.dataframe(pd.DataFrame([student_data]), use_container_width=True)

# Model info sidebar - FIXED
with st.sidebar.expander("ℹ️ Model Information"):
    st.info(f"""
    **Logistic Regression Model**  
    ✅ Accuracy: {model_data['accuracy']:.1%}  
    ✅ Features: {len(model_data['features'])}  
    ✅ Polynomial Features: Enhanced (degree=2)  
    ✅ SMOTE: Balanced classes  
    ✅ Test Score: {model_data['accuracy']:.4f}  
    """)
    
    st.caption("Trained on student performance dataset")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "Built with ❤️ for B.Tech Student | SVECW Bhimavaram</p>", 
    unsafe_allow_html=True
)
