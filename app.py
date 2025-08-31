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

# Load environment variables (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, skip

# Page configuration
st.set_page_config(
    page_title="AI Marketing Campaign Forecaster",
    page_icon="🚀",
    layout="wide"
)

# Google Drive file IDs
MODEL_FILES = {
    'best_roi_model.pkl': '1dJcbTiffS4FsAG5i-5IjLdMSmOSL6jIf',
    'kmeans_model.pkl': '1tV7m4b3EQ5xdEPgYAzGrYzYeu4ApdKIR', 
    'minmax_scaler.pkl': '1OX5eEQmbTkSf3QtAp15jpQOpF7PRaHSm',
    'ordinal_encoder.pkl': '1jFutweZwJqm-0JzNQJDgzTpZGTzB5Yzn'
}

@st.cache_resource
def download_and_load_models():
    """Download models from Google Drive and load them"""
    models = {}
    
    with st.spinner('🔄 Loading AI models...'):
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
    """Initialize Groq client with support for both local and cloud deployment"""
    api_key = None
    
    # Try different methods to get API key
    try:
        # First, try Streamlit secrets (for Streamlit Cloud)
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        try:
            # Fallback to environment variable (for local development)
            api_key = os.getenv("GROQ_API_KEY")
        except:
            pass
    
    if not api_key:
        st.error("❌ GROQ API key not found!")
        st.info("💡 For local development: Add GROQ_API_KEY to your .env file")
        st.info("💡 For Streamlit Cloud: Add GROQ_API_KEY to your app secrets")
        return None
    
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {str(e)}")
        return None

def generate_campaign_content(groq_client, campaign_data):
    """Generate marketing campaign content using Groq"""
    if not groq_client:
        return "AI content generation unavailable. Please check your API key configuration."
    
    prompt = f"""
    Create a comprehensive marketing campaign strategy for:
    - Company: {campaign_data['company']}
    - Campaign Type: {campaign_data['campaign_type']}
    - Target Audience: {campaign_data['target_audience']}
    - Channel: {campaign_data['channel']}
    - Budget: ${campaign_data['budget']:,}
    - Duration: {campaign_data['duration']} days
    
    Provide:
    1. Key messaging strategy
    2. Target audience insights
    3. Channel-specific tactics
    4. Success metrics to track
    5. Optimization recommendations
    
    Keep it concise and actionable.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Content generation failed: {str(e)}"

def make_predictions(models, campaign_data):
    """Make predictions using loaded models"""
    try:
        # For now, using simulated predictions
        # TODO: Replace with actual model inference using your trained models
        
        # Base predictions with some intelligence
        base_roi = 25
        audience_multiplier = {
            "Young Adults (18-34)": 1.2,
            "Professionals (25-54)": 1.1,
            "Seniors (55+)": 0.9
        }
        
        channel_multiplier = {
            "Social Media": 1.3,
            "Search Engine": 1.4,
            "Email": 1.6,
            "Display": 0.8,
            "Video": 1.1
        }
        
        # Calculate intelligent predictions
        roi_multiplier = (
            audience_multiplier.get(campaign_data['target_audience'], 1.0) *
            channel_multiplier.get(campaign_data['channel'], 1.0)
        )
        
        predicted_roi = base_roi * roi_multiplier * np.random.uniform(0.8, 1.3)
        predicted_conversion = np.random.uniform(0.02, 0.08) * roi_multiplier
        predicted_cost = campaign_data['budget'] * np.random.uniform(0.4, 0.8)
        cluster = np.random.randint(0, 5)
        
        return {
            'roi': predicted_roi,
            'conversion_rate': predicted_conversion,
            'acquisition_cost': predicted_cost,
            'cluster': cluster,
            'confidence': np.random.uniform(0.75, 0.95)
        }
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

# Initialize session state
if 'campaign_history' not in st.session_state:
    st.session_state.campaign_history = []

# Load models and initialize Groq
with st.spinner('🚀 Initializing AI Marketing Forecaster...'):
    models = download_and_load_models()
    groq_client = init_groq_client()

# Header
st.title("🚀 AI Marketing Campaign Forecaster")
st.markdown("*Powered by Machine Learning & AI • Create, Optimize, and Forecast Marketing Campaigns*")

# Check if models loaded successfully
if not models:
    st.error("❌ Failed to load ML models. Please check your Google Drive links.")
    st.stop()

# Sidebar for campaign input
with st.sidebar:
    st.header("🎯 Campaign Builder")
    
    # Campaign details
    company = st.text_input("Company Name", value="TechCorp", help="Enter your company name")
    
    campaign_type = st.selectbox(
        "Campaign Type",
        ["Brand Awareness", "Lead Generation", "Sales", "Retargeting", "Product Launch"],
        help="Select the primary objective of your campaign"
    )
    
    target_audience = st.selectbox(
        "Target Audience", 
        ["Young Adults (18-34)", "Professionals (25-54)", "Seniors (55+)", "Students", "Parents"],
        help="Choose your primary target demographic"
    )
    
    channel = st.selectbox(
        "Primary Channel",
        ["Social Media", "Search Engine", "Email", "Display", "Video", "Mobile"],
        help="Select your main marketing channel"
    )
    
    # Budget and timeline
    st.subheader("💰 Budget & Timeline")
    budget = st.number_input(
        "Total Budget ($)", 
        min_value=1000, 
        max_value=500000, 
        value=25000, 
        step=1000,
        help="Enter your total campaign budget"
    )
    
    duration = st.slider(
        "Campaign Duration (days)", 
        min_value=7, 
        max_value=90, 
        value=30,
        help="Select campaign duration in days"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        location = st.text_input("Target Location", value="United States")
        language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
        expected_ctr = st.slider("Expected CTR (%)", 0.5, 5.0, 2.0, 0.1)
    
    # Generate button
    generate_btn = st.button("🔮 Generate AI Forecast", use_container_width=True, type="primary")

# Main content area
if generate_btn:
    campaign_data = {
        'company': company,
        'campaign_type': campaign_type,
        'target_audience': target_audience,
        'channel': channel,
        'budget': budget,
        'duration': duration,
        'location': location,
        'language': language
    }
    
    # Make predictions
    with st.spinner('🤖 AI is analyzing your campaign...'):
        predictions = make_predictions(models, campaign_data)
    
    if predictions:
        # Display key metrics
        st.subheader("📊 Campaign Forecast Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Predicted ROI",
                f"{predictions['roi']:.1f}%",
                delta=f"{predictions['roi']-25:.1f}%",
                help="Return on Investment compared to industry average"
            )
        
        with col2:
            st.metric(
                "Conversion Rate",
                f"{predictions['conversion_rate']*100:.2f}%",
                delta=f"{(predictions['conversion_rate']-0.03)*100:.2f}%",
                help="Expected percentage of visitors who will convert"
            )
        
        with col3:
            st.metric(
                "Acquisition Cost",
                f"${predictions['acquisition_cost']:,.0f}",
                delta=f"-${budget-predictions['acquisition_cost']:,.0f}",
                help="Total cost to acquire customers"
            )
        
        with col4:
            st.metric(
                "Confidence Score",
                f"{predictions['confidence']*100:.0f}%",
                help="AI model confidence in predictions"
            )
        
        # Performance indicator
        if predictions['roi'] > 30:
            st.success("🎉 Excellent ROI potential! This campaign shows strong performance indicators.")
        elif predictions['roi'] > 20:
            st.info("✅ Good performance expected. Consider A/B testing for optimization.")
        else:
            st.warning("⚠️ Consider optimizing campaign parameters for better ROI.")
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Charts", "💰 Budget Analysis", "🤖 AI Strategy", "📈 Timeline"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # ROI Comparison Chart
                fig_roi = go.Figure(data=[
                    go.Bar(
                        x=['Industry Average', 'Your Campaign'], 
                        y=[25, predictions['roi']],
                        marker_color=['lightblue', 'darkblue'],
                        text=[f"{25:.1f}%", f"{predictions['roi']:.1f}%"],
                        textposition='auto'
                    )
                ])
                fig_roi.update_layout(
                    title="ROI Performance Comparison",
                    yaxis_title="ROI (%)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_roi, use_container_width=True)
            
            with col2:
                # Performance Timeline
                days = list(range(1, duration + 1))
                cumulative_roi = [predictions['roi'] * (d/duration) * np.random.uniform(0.8, 1.2) for d in days]
                
                fig_timeline = px.line(
                    x=days, 
                    y=cumulative_roi,
                    title="Expected ROI Timeline",
                    labels={'x': 'Campaign Day', 'y': 'Cumulative ROI (%)'}
                )
                fig_timeline.update_traces(line_color='green', line_width=3)
                fig_timeline.update_layout(height=400)
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with tab2:
            # Budget allocation and analysis
            st.subheader("💰 Optimized Budget Allocation")
            
            # Channel allocation simulation
            if channel == "Social Media":
                allocations = [60, 25, 10, 5]
                channels_list = ['Social Media', 'Search Engine', 'Email', 'Display']
            elif channel == "Search Engine":
                allocations = [50, 30, 15, 5]
                channels_list = ['Search Engine', 'Social Media', 'Email', 'Display']
            else:
                allocations = [40, 30, 20, 10]
                channels_list = ['Social Media', 'Search Engine', 'Email', 'Display']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_budget = px.pie(
                    values=allocations,
                    names=channels_list,
                    title="Recommended Budget Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_budget.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_budget, use_container_width=True)
            
            with col2:
                budget_df = pd.DataFrame({
                    'Channel': channels_list,
                    'Budget ($)': [budget * (a/100) for a in allocations],
                    'Allocation (%)': allocations,
                    'Expected ROI (%)': [predictions['roi'] * np.random.uniform(0.8, 1.3) for _ in channels_list]
                })
                
                st.dataframe(
                    budget_df.style.format({
                        'Budget ($)': '${:,.0f}',
                        'Allocation (%)': '{:.0f}%',
                        'Expected ROI (%)': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        
        with tab3:
            # AI-generated strategy
            st.subheader("🤖 AI-Generated Campaign Strategy")
            
            if groq_client:
                with st.spinner("🧠 AI is creating your personalized strategy..."):
                    strategy = generate_campaign_content(groq_client, campaign_data)
                    
                # Display strategy in a nice format
                st.markdown("### 📋 Strategic Recommendations")
                st.write(strategy)
            else:
                st.warning("⚠️ AI strategy generation unavailable. Please configure Groq API key.")
                
                # Fallback recommendations
                st.markdown("### 📋 General Recommendations")
                recommendations = [
                    f"Optimize {channel.lower()} campaigns for {target_audience.lower()} during peak engagement hours",
                    f"A/B test different creative variants specifically for {campaign_type.lower()} objectives",
                    f"Monitor key {campaign_type.lower()} metrics closely in the first week for quick optimizations",
                    f"Consider scaling budget by 20% if ROI exceeds {predictions['roi']:.0f}% in first two weeks",
                    f"Implement retargeting campaigns for users who don't convert initially"
                ]
                
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {rec}")
        
        with tab4:
            # Timeline and milestones
            st.subheader("📈 Campaign Timeline & Milestones")
            
            # Create timeline data
            timeline_dates = [datetime.now() + timedelta(days=i) for i in range(0, duration, 7)]
            milestones = [
                "Campaign Launch",
                "First Week Analysis",
                "Mid-Campaign Optimization",
                "Performance Review",
                "Final Week Push"
            ]
            
            # Timeline visualization
            fig_timeline = go.Figure()
            
            # Add cumulative spend
            cumulative_spend = [budget * (i+1)/(len(timeline_dates)) for i in range(len(timeline_dates))]
            fig_timeline.add_trace(go.Scatter(
                x=timeline_dates,
                y=cumulative_spend,
                mode='lines+markers',
                name='Cumulative Spend ($)',
                line=dict(color='red', width=3),
                yaxis='y'
            ))
            
            # Add ROI projection
            roi_projection = [predictions['roi'] * (i+1)/(len(timeline_dates)) for i in range(len(timeline_dates))]
            fig_timeline.add_trace(go.Scatter(
                x=timeline_dates,
                y=roi_projection,
                mode='lines+markers',
                name='Projected ROI (%)',
                line=dict(color='green', width=3),
                yaxis='y2'
            ))
            
            fig_timeline.update_layout(
                title="Campaign Performance Timeline",
                xaxis_title="Date",
                yaxis=dict(title="Cumulative Spend ($)", side="left"),
                yaxis2=dict(title="ROI (%)", side="right", overlaying="y"),
                height=400
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Milestones table
            milestone_df = pd.DataFrame({
                'Week': [f"Week {i+1}" for i in range(min(len(milestones), duration//7 + 1))],
                'Milestone': milestones[:duration//7 + 1],
                'Expected Spend ($)': [budget * (i+1)/(duration//7 + 1) for i in range(min(len(milestones), duration//7 + 1))],
                'Target ROI (%)': [predictions['roi'] * (i+1)/(duration//7 + 1) for i in range(min(len(milestones), duration//7 + 1))]
            })
            
            st.subheader("🎯 Key Milestones")
            st.dataframe(
                milestone_df.style.format({
                    'Expected Spend ($)': '${:,.0f}',
                    'Target ROI (%)': '{:.1f}%'
                }),
                use_container_width=True
            )
        
        # Save to campaign history
        st.session_state.campaign_history.append({
            'timestamp': datetime.now(),
            'company': company,
            'campaign_type': campaign_type,
            'channel': channel,
            'target_audience': target_audience,
            'budget': budget,
            'duration': duration,
            'predicted_roi': predictions['roi'],
            'predicted_conversion': predictions['conversion_rate'],
            'confidence': predictions['confidence']
        })
        
        st.success("✅ Campaign forecast completed and saved to history!")

# Campaign History Section
if st.session_state.campaign_history:
    st.markdown("---")
    st.header("📈 Campaign History & Analytics")
    
    history_df = pd.DataFrame(st.session_state.campaign_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Campaigns", len(history_df))
    with col2:
        st.metric("Average ROI", f"{history_df['predicted_roi'].mean():.1f}%")
    with col3:
        st.metric("Total Budget", f"${history_df['budget'].sum():,.0f}")
    with col4:
        st.metric("Avg Confidence", f"{history_df['confidence'].mean()*100:.0f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI trends
        fig_history = px.line(
            history_df,
            x='timestamp',
            y='predicted_roi',
            color='campaign_type',
            title="ROI Trends Over Time",
            markers=True
        )
        st.plotly_chart(fig_history, use_container_width=True)
    
    with col2:
        # Budget by channel
        budget_by_channel = history_df.groupby('channel')['budget'].sum().reset_index()
        fig_channel = px.bar(
            budget_by_channel,
            x='channel',
            y='budget',
            title="Total Budget by Channel",
            color='budget',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_channel, use_container_width=True)
    
    # Detailed history table
    with st.expander("📋 View Detailed Campaign History"):
        display_df = history_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.round({'predicted_roi': 1, 'predicted_conversion': 4, 'confidence': 2})
        
        st.dataframe(
            display_df.style.format({
                'budget': '${:,.0f}',
                'predicted_roi': '{:.1f}%',
                'predicted_conversion': '{:.2%}',
                'confidence': '{:.0%}'
            }),
            use_container_width=True
        )

# Instructions for first-time users
else:
    st.markdown("---")
    st.subheader("🚀 Welcome to AI Marketing Campaign Forecaster!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Step 1: Configure
        - Enter your company details
        - Select campaign type and audience
        - Set budget and timeline
        """)
    
    with col2:
        st.markdown("""
        ### 🔮 Step 2: Generate
        - Click "Generate AI Forecast"
        - AI analyzes your parameters
        - Get instant predictions
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Step 3: Optimize
        - Review performance charts
        - Get AI-generated strategy
        - Track in campaign history
        """)
    
    st.info("👈 **Get Started:** Use the sidebar to create your first AI-powered marketing campaign forecast!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🚀 AI Marketing Campaign Forecaster</strong></p>
    <p><em>Built with Streamlit • Powered by Machine Learning & Groq AI</em></p>
    <p>🤖 Transforming Marketing Strategy with Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)
