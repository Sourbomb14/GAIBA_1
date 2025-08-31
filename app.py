# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gdown
import joblib
import os
from groq import Groq
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Marketing Campaign Forecaster",
    page_icon="🚀",
    layout="wide"
)

# Google Drive file IDs (replace with your actual file IDs)
MODEL_FILES = {
    'best_roi_model.pkl': 'your_roi_model_drive_id',
    'kmeans_model.pkl': 'your_kmeans_model_drive_id', 
    'minmax_scaler.pkl': 'your_scaler_drive_id',
    'ordinal_encoder.pkl': 'your_encoder_drive_id'
}

@st.cache_resource
def download_and_load_models():
    """Download models from Google Drive and load them"""
    models = {}
    
    with st.spinner('Loading AI models...'):
        for filename, file_id in MODEL_FILES.items():
            if not os.path.exists(filename):
                try:
                    # Download from Google Drive
                    url = f'https://drive.google.com/uc?id={file_id}'
                    gdown.download(url, filename, quiet=True)
                    st.success(f"✅ Downloaded {filename}")
                except Exception as e:
                    st.error(f"❌ Failed to download {filename}: {str(e)}")
                    return None
            
            # Load the model
            try:
                models[filename.replace('.pkl', '')] = joblib.load(filename)
            except Exception as e:
                st.error(f"❌ Failed to load {filename}: {str(e)}")
                return None
    
    return models

@st.cache_resource
def init_groq_client():
    """Initialize Groq client"""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ API key not found in secrets!")
        return None
    return Groq(api_key=api_key)

def generate_campaign_content(groq_client, campaign_data):
    """Generate marketing campaign content using Groq"""
    if not groq_client:
        return "AI content generation unavailable."
    
    prompt = f"""
    Create a marketing campaign strategy for:
    - Company: {campaign_data['company']}
    - Campaign Type: {campaign_data['campaign_type']}
    - Target Audience: {campaign_data['target_audience']}
    - Channel: {campaign_data['channel']}
    - Budget: ${campaign_data['budget']:,}
    - Duration: {campaign_data['duration']} days
    
    Provide a brief strategy with key messaging points and tactics.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Content generation failed: {str(e)}"

def make_predictions(models, campaign_data):
    """Make predictions using loaded models"""
    try:
        # Simulate predictions (replace with actual model inference)
        predicted_roi = np.random.uniform(15, 45)
        predicted_conversion = np.random.uniform(0.02, 0.08)
        predicted_cost = campaign_data['budget'] * np.random.uniform(0.4, 0.8)
        cluster = np.random.randint(0, 5)
        
        return {
            'roi': predicted_roi,
            'conversion_rate': predicted_conversion,
            'acquisition_cost': predicted_cost,
            'cluster': cluster,
            'confidence': np.random.uniform(0.7, 0.95)
        }
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Initialize session state
if 'campaign_history' not in st.session_state:
    st.session_state.campaign_history = []

# Load models and initialize Groq
models = download_and_load_models()
groq_client = init_groq_client()

# Main App
st.title("🚀 AI Marketing Campaign Forecaster")
st.markdown("*Powered by AI • Create, Optimize, and Forecast Marketing Campaigns*")

# Sidebar
with st.sidebar:
    st.header("🎯 Campaign Builder")
    
    company = st.text_input("Company Name", value="TechCorp")
    
    campaign_type = st.selectbox(
        "Campaign Type",
        ["Brand Awareness", "Lead Generation", "Sales", "Retargeting"]
    )
    
    target_audience = st.selectbox(
        "Target Audience", 
        ["Young Adults (18-34)", "Professionals (25-54)", "Seniors (55+)"]
    )
    
    channel = st.selectbox(
        "Primary Channel",
        ["Social Media", "Search Engine", "Email", "Display", "Video"]
    )
    
    budget = st.number_input(
        "Budget ($)", 
        min_value=1000, 
        max_value=500000, 
        value=25000, 
        step=1000
    )
    
    duration = st.slider("Duration (days)", 7, 90, 30)
    
    generate_btn = st.button("🔮 Generate Campaign", use_container_width=True)

# Main content area
if generate_btn and models:
    campaign_data = {
        'company': company,
        'campaign_type': campaign_type,
        'target_audience': target_audience,
        'channel': channel,
        'budget': budget,
        'duration': duration
    }
    
    # Make predictions
    predictions = make_predictions(models, campaign_data)
    
    if predictions:
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Predicted ROI",
                f"{predictions['roi']:.1f}%",
                delta=f"{predictions['roi']-25:.1f}%"
            )
        
        with col2:
            st.metric(
                "Conversion Rate",
                f"{predictions['conversion_rate']*100:.2f}%",
                delta=f"{(predictions['conversion_rate']-0.03)*100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Acquisition Cost",
                f"${predictions['acquisition_cost']:,.0f}",
                delta=f"-${budget-predictions['acquisition_cost']:,.0f}"
            )
        
        with col4:
            st.metric(
                "Confidence Score",
                f"{predictions['confidence']*100:.0f}%"
            )
        
        # Performance indicator
        if predictions['roi'] > 30:
            st.success("🎉 Excellent ROI potential!")
        elif predictions['roi'] > 20:
            st.info("✅ Good performance expected")
        else:
            st.warning("⚠️ Consider optimization")
        
        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["📊 Performance Charts", "💰 Budget Analysis", "🤖 AI Campaign Strategy"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # ROI Comparison
                fig_roi = go.Figure(data=[
                    go.Bar(x=['Industry Avg', 'Your Campaign'], 
                          y=[25, predictions['roi']],
                          marker_color=['lightblue', 'darkblue'])
                ])
                fig_roi.update_layout(title="ROI Comparison", yaxis_title="ROI (%)")
                st.plotly_chart(fig_roi, use_container_width=True)
            
            with col2:
                # Performance Timeline
                days = list(range(1, duration + 1))
                cumulative_roi = [predictions['roi'] * (d/duration) for d in days]
                
                fig_timeline = px.line(
                    x=days, 
                    y=cumulative_roi,
                    title="Expected ROI Timeline",
                    labels={'x': 'Day', 'y': 'Cumulative ROI (%)'}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with tab2:
            # Budget allocation simulation
            channels = ['Social Media', 'Search Engine', 'Email', 'Display']
            allocations = [40, 30, 20, 10]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_budget = px.pie(
                    values=allocations,
                    names=channels,
                    title="Recommended Budget Distribution"
                )
                st.plotly_chart(fig_budget, use_container_width=True)
            
            with col2:
                budget_df = pd.DataFrame({
                    'Channel': channels,
                    'Budget ($)': [budget * (a/100) for a in allocations],
                    'Expected ROI (%)': [np.random.uniform(20, 40) for _ in channels]
                })
                st.dataframe(budget_df, use_container_width=True)
        
        with tab3:
            st.subheader("🤖 AI-Generated Campaign Strategy")
            
            with st.spinner("Generating AI strategy..."):
                strategy = generate_campaign_content(groq_client, campaign_data)
                st.write(strategy)
            
            st.subheader("📋 Key Recommendations")
            recommendations = [
                f"Optimize for {channel.lower()} engagement during peak hours",
                f"A/B test creative variants for {target_audience.lower()}",
                f"Monitor {campaign_type.lower()} metrics closely in first week",
                f"Consider scaling budget if ROI exceeds {predictions['roi']:.0f}%"
            ]
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # Save to history
        st.session_state.campaign_history.append({
            'timestamp': datetime.now(),
            'company': company,
            'campaign_type': campaign_type,
            'budget': budget,
            'predicted_roi': predictions['roi'],
            'predicted_conversion': predictions['conversion_rate']
        })

# Campaign History
if st.session_state.campaign_history:
    st.markdown("---")
    st.subheader("📈 Campaign History")
    
    history_df = pd.DataFrame(st.session_state.campaign_history)
    history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Campaigns", len(history_df))
    with col2:
        st.metric("Avg ROI", f"{history_df['predicted_roi'].mean():.1f}%")
    with col3:
        st.metric("Total Budget", f"${history_df['budget'].sum():,.0f}")
    
    # History chart
    fig_history = px.line(
        history_df,
        x='timestamp',
        y='predicted_roi',
        color='campaign_type',
        title="ROI Trends Over Time"
    )
    st.plotly_chart(fig_history, use_container_width=True)
    
    # Data table
    st.dataframe(history_df, use_container_width=True)

elif not models:
    st.error("❌ Models failed to load. Please check your Google Drive file IDs.")
else:
    st.info("👈 Use the sidebar to create your first AI-powered marketing campaign!")

# Footer
st.markdown("---")
st.markdown("**🚀 AI Marketing Forecaster** | *Built with Streamlit & Groq AI*")
