import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import json
import io
import chardet
import random

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="🎯 AI Marketing Campaign Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Dark Theme CSS
st.markdown("""
<style>
    /* Dark theme variables */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #1e1e2e;
        --bg-tertiary: #262730;
        --text-primary: #ffffff;
        --text-secondary: #b3b3b3;
        --accent-primary: #667eea;
        --accent-secondary: #764ba2;
        --success: #00d4aa;
        --warning: #ff9500;
        --error: #ff4757;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: var(--text-primary);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        color: var(--text-primary);
        backdrop-filter: blur(10px);
    }
    
    .cluster-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 25%, #4facfe 50%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1e2e 0%, #262730 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #00d4aa);
        border-radius: 10px;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Success animation */
    .success-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(0, 212, 170, 0.5); }
        50% { transform: scale(1.02); box-shadow: 0 0 30px rgba(0, 212, 170, 0.8); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(0, 212, 170, 0.5); }
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: var(--bg-secondary);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* API Key input styling */
    .api-key-container {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 100, 100, 0.3);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def detect_csv_separator(file):
    """Auto-detect CSV separator"""
    try:
        # Read a sample of the file
        sample = file.read(1024).decode('utf-8')
        file.seek(0)  # Reset file pointer
        
        # Count common separators
        separators = [',', ';', '\t', '|']
        separator_counts = {}
        
        for sep in separators:
            separator_counts[sep] = sample.count(sep)
        
        # Return the most common separator
        best_separator = max(separator_counts, key=separator_counts.get)
        return best_separator if separator_counts[best_separator] > 0 else ','
    except:
        return ','

@st.cache_data
def load_data_with_encoding(file, file_type):
    """Load data with automatic encoding detection"""
    try:
        if file_type == 'csv':
            # Detect encoding
            raw_data = file.read()
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result['encoding'] if encoding_result['encoding'] else 'utf-8'
            file.seek(0)
            
            # Detect separator
            separator = detect_csv_separator(file)
            
            # Load with detected encoding and separator
            df = pd.read_csv(io.StringIO(raw_data.decode(encoding)), sep=separator)
            return df, f"Detected separator: '{separator}', Encoding: {encoding}"
        
        elif file_type == 'xlsx':
            df = pd.read_excel(file)
            return df, "Excel file loaded successfully"
        
        elif file_type == 'json':
            df = pd.read_json(file)
            return df, "JSON file loaded successfully"
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, str(e)

# --- Initialize OpenAI Client ---
@st.cache_resource
def initialize_openai(_api_key):
    """Initialize OpenAI client with provided API key"""
    if not _api_key:
        return None
    try:
        client = OpenAI(api_key=_api_key)
        # Test the connection with a simple call
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return client
    except Exception as e:
        st.error(f"❌ Invalid API key or connection error: {str(e)}")
        return None

# --- Header Section ---
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Marketing Campaign Generator</h1>
    <p>Transform your customer data into personalized marketing campaigns with AI-powered insights</p>
    <div style="margin-top: 1rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
            ✨ Auto CSV Detection | 🔒 Secure | 🤖 AI-Powered
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔧 Campaign Settings")
    
    # --- API Key Management Section ---
    st.markdown("""
    <div class="api-key-container">
        <h4>🔐 OpenAI API Configuration</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to get API key from environment first
    api_key = os.getenv("OPENAI_API_KEY")
    client = None
    
    if api_key:
        # API key found in environment
        st.success("✅ API Key loaded from environment")
        client = initialize_openai(api_key)
        if client:
            st.success("🔗 OpenAI Connected")
        else:
            st.error("❌ Environment API key is invalid")
            api_key = None
    
    if not api_key:
        # No environment key or invalid - show input field
        st.markdown("**Enter your OpenAI API Key:**")
        user_api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="sk-...",
            help="Get your API key from https://platform.openai.com/api-keys",
            label_visibility="collapsed"
        )
        
        if user_api_key:
            # Test the user-provided key
            with st.spinner("🔍 Validating API key..."):
                client = initialize_openai(user_api_key)
                if client:
                    api_key = user_api_key
                    st.success("✅ API Key validated successfully!")
                    st.success("🔗 OpenAI Connected")
                    # Store in session state
                    st.session_state['api_key'] = user_api_key
                else:
                    st.error("❌ Invalid API key. Please check and try again.")
        else:
            st.warning("⚠️ Please enter your OpenAI API key to continue")
            st.info("💡 Your API key is only used for this session and is not stored permanently.")
    
    # Show connection status
    if not client:
        st.error("❌ OpenAI Not Connected")
        st.markdown("""
        <div style="background: rgba(255, 100, 100, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(255, 100, 100, 0.3); margin: 1rem 0;">
            <small><strong>🔒 Need an API key?</strong><br>
            1. Go to <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI API Keys</a><br>
            2. Create a new secret key<br>
            3. Copy and paste it above</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Campaign configuration (only show if connected)
    if client:
        campaign_goal = st.selectbox(
            "🎯 Campaign Objective",
            ["Increase Sales", "Boost Engagement", "Promote New Product", "Customer Retention", "Brand Awareness", "Lead Generation"],
            index=1
        )
        
        custom_goal = st.text_input("Or enter custom goal:", placeholder="e.g., Promote holiday season offers")
        if custom_goal:
            campaign_goal = custom_goal
        
        st.divider()
        
        # Advanced settings
        st.subheader("⚙️ Advanced Settings")
        
        num_groups = st.slider(
            "🎭 Number of Target Groups",
            min_value=2,
            max_value=10,
            value=3,
            help="More groups = more personalized campaigns"
        )
        
        model_choice = st.selectbox(
            "🤖 AI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0  # Default to gpt-3.5-turbo for cost efficiency
        )
        
        # Message generation settings
        messages_per_group = st.slider(
            "💌 AI Messages per Group",
            min_value=3,
            max_value=20,
            value=5,
            help="Number of unique AI messages to generate per segment"
        )
        
        # Add debug mode
        debug_mode = st.checkbox("🔧 Debug Mode", help="Show API call details")
        
        # Total messages info
        total_ai_messages = num_groups * messages_per_group
        st.info(f"💡 Will generate {total_ai_messages} unique AI messages (cost: ~${total_ai_messages * 0.002:.3f})")
    else:
        st.info("🔒 Connect your OpenAI API key to access campaign settings")

# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose your customer data file",
        type=["csv", "xlsx", "json"],
        help="Upload CSV, Excel, or JSON files with customer data. CSV separators are auto-detected."
    )

with col2:
    if uploaded_file:
        st.success("✅ File uploaded!")
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")

# --- Data Processing ---
if uploaded_file is not None and client is not None:
    try:
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Load data with auto-detection
        df, load_info = load_data_with_encoding(uploaded_file, file_extension)
        
        if df is not None:
            st.success(f"✅ {load_info}")
            
            # Data overview
            st.header("📊 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📋 Total Records", len(df))
            with col2:
                st.metric("📈 Features", len(df.columns))
            with col3:
                st.metric("🔍 Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("📊 Data Types", df.dtypes.nunique())
            
            # Data preview with search and filter
            st.subheader("🔍 Data Preview")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("🔎 Search in data:", placeholder="Enter search term...")
            with col2:
                show_nulls = st.checkbox("Show missing values only", value=False)
            
            # Filter data based on search and null values
            display_df = df.copy()
            if search_term:
                mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                display_df = display_df[mask]
            
            if show_nulls:
                display_df = display_df[display_df.isnull().any(axis=1)]
            
            st.dataframe(display_df.head(20), use_container_width=True, height=400)
            
            # Feature selection for clustering
            st.header("🎯 Feature Selection")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Numerical Features")
                numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_numeric = st.multiselect(
                    "Select numerical features:",
                    numeric_features,
                    default=numeric_features[:3] if len(numeric_features) >= 3 else numeric_features
                )
            
            with col2:
                st.subheader("Categorical Features")
                categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
                selected_categorical = st.multiselect(
                    "Select categorical features:",
                    categorical_features,
                    default=categorical_features[:2] if len(categorical_features) >= 2 else categorical_features
                )
            
            all_selected_features = selected_numeric + selected_categorical
            
            if not all_selected_features:
                st.warning("⚠️ Please select at least one feature for clustering.")
            else:
                st.success(f"✅ Selected {len(all_selected_features)} features for analysis")
                
                # Feature importance preview
                if selected_numeric:
                    st.subheader("📊 Feature Distribution")
                    fig = make_subplots(
                        rows=1, cols=min(3, len(selected_numeric)),
                        subplot_titles=selected_numeric[:3]
                    )
                    
                    for i, feature in enumerate(selected_numeric[:3]):
                        fig.add_trace(
                            go.Histogram(x=df[feature], name=feature, showlegend=False),
                            row=1, col=i+1
                        )
                    
                    fig.update_layout(
                        height=300, 
                        title_text="Distribution of Selected Numerical Features",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Campaign Generation Button
            if st.button("🚀 Generate AI Marketing Campaign", type="primary", use_container_width=True):
                if not all_selected_features:
                    st.error("❌ Please select features for clustering first!")
                else:
                    # Progress tracking
                    progress_container = st.container()
                    with progress_container:
                        st.markdown("""
                        <div class="success-animation" style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%); border-radius: 15px; margin: 1rem 0; border: 1px solid rgba(102, 126, 234, 0.5);">
                            <h2>🔄 Processing Your Campaign...</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        main_progress = st.progress(0, text="Initializing...")
                        status_text = st.empty()
                        
                        # Step 1: Data Preprocessing
                        status_text.text("🔧 Preprocessing data...")
                        main_progress.progress(10, text="Preprocessing data...")
                        
                        # Prepare features for clustering
                        X = df[all_selected_features].copy()
                        
                        # Handle missing values
                        for col in selected_numeric:
                            X[col] = X[col].fillna(X[col].median())
                        
                        for col in selected_categorical:
                            X[col] = X[col].fillna("Unknown")
                        
                        # One-hot encode categorical variables
                        if selected_categorical:
                            X_processed = pd.get_dummies(X, columns=selected_categorical, drop_first=True)
                        else:
                            X_processed = X.copy()
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_processed)
                        
                        time.sleep(1)
                        
                        # Step 2: Clustering
                        status_text.text("🎯 Creating customer segments...")
                        main_progress.progress(30, text="Creating customer segments...")
                        
                        # Perform clustering
                        kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        df['Cluster'] = cluster_labels
                        
                        # PCA for visualization
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        df['PCA1'] = X_pca[:, 0]
                        df['PCA2'] = X_pca[:, 1]
                        
                        time.sleep(1)
                        main_progress.progress(50, text="Analyzing clusters...")
                        
                        # Step 3: Cluster Analysis
                        status_text.text("📊 Analyzing customer segments...")
                        
                        cluster_info = {}
                        cluster_stats = []
                        
                        for cluster_id in range(num_groups):
                            cluster_data = df[df['Cluster'] == cluster_id]
                            cluster_size = len(cluster_data)
                            
                            if cluster_size == 0:
                                continue
                            
                            # Generate cluster insights
                            sample_data = cluster_data[all_selected_features].head(5)
                            
                            # Create a more focused description
                            cluster_description = []
                            for feature in all_selected_features:
                                if feature in selected_numeric:
                                    mean_val = cluster_data[feature].mean()
                                    cluster_description.append(f"{feature}: average {mean_val:.2f}")
                                else:
                                    top_value = cluster_data[feature].mode().iloc[0] if len(cluster_data[feature].mode()) > 0 else "Mixed"
                                    cluster_description.append(f"{feature}: mostly {top_value}")
                            
                            description_text = ", ".join(cluster_description)
                            
                            # AI-generated cluster name and insights
                            cluster_prompt = f"""
                            Analyze this customer segment for campaign: "{campaign_goal}"

                            Segment size: {cluster_size} customers
                            Key characteristics: {description_text}
                            
                            Sample data:
                            {sample_data.to_string(index=False)}
                            
                            Provide ONLY a JSON response:
                            {{
                                "name": "2-3 word catchy segment name",
                                "slogan": "One compelling sentence for marketing",
                                "characteristics": ["trait 1", "trait 2", "trait 3"]
                            }}
                            """
                            
                            try:
                                response = client.chat.completions.create(
                                    model=model_choice,
                                    messages=[{"role": "user", "content": cluster_prompt}],
                                    max_tokens=150,
                                    temperature=0.7
                                )
                                
                                # Clean the response to ensure it's valid JSON
                                response_text = response.choices[0].message.content.strip()
                                # Remove any markdown formatting
                                if response_text.startswith('```json'):
                                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                                elif response_text.startswith('```'):
                                    response_text = response_text.replace('```', '').strip()
                                
                                cluster_analysis = json.loads(response_text)
                                
                                # Validate the response has required keys
                                if not all(key in cluster_analysis for key in ['name', 'slogan', 'characteristics']):
                                    raise ValueError("Missing required keys in response")
                                
                            except Exception as e:
                                if debug_mode:
                                    st.warning(f"Cluster analysis failed for cluster {cluster_id + 1}: {str(e)}")
                                cluster_analysis = {
                                    "name": f"Group {cluster_id + 1}",
                                    "slogan": f"Targeted approach for {campaign_goal.lower()}",
                                    "characteristics": ["Data-driven insights", "Targeted messaging", "Customer-focused approach"]
                                }
                            
                            cluster_info[cluster_id] = cluster_analysis
                            cluster_stats.append({
                                'Cluster': cluster_id,
                                'Name': cluster_analysis['name'],
                                'Size': cluster_size,
                                'Percentage': f"{cluster_size/len(df)*100:.1f}%"
                            })
                        
                        main_progress.progress(70, text="Generating personalized messages...")
                        
                        # Step 4: Enhanced Message Generation with Unique Content
                        status_text.text("💌 Creating unique AI messages...")
                        
                        # Enhanced message generation function
                        def generate_unique_message(customer_idx, customer_data, cluster_details, campaign_goal, message_number):
                            """Generate a unique personalized message with variation"""
                            
                            # Create customer profile with specific details
                            profile_details = []
                            for feature, value in customer_data.items():
                                if feature not in ['Cluster', 'PCA1', 'PCA2', 'Cluster_Name', 'Cluster_Slogan', 'Personalized_Message']:
                                    if pd.notna(value):
                                        profile_details.append(f"{feature}: {value}")
                            
                            profile_text = ", ".join(profile_details[:4])  # Limit to avoid token overflow
                            
                            # Add variation to prompts for uniqueness
                            greeting_styles = ["Hi", "Hello", "Hey there", "Greetings", "Dear customer"]
                            cta_styles = ["Don't miss out!", "Act now!", "Limited time offer!", "Exclusive for you!", "Get started today!"]
                            
                            greeting = random.choice(greeting_styles)
                            cta = random.choice(cta_styles)
                            
                            # Enhanced prompt with more specific instructions
                            prompt = f"""
                            Create a personalized marketing message #{message_number} for a customer in the "{cluster_details['name']}" segment.

                            CAMPAIGN GOAL: {campaign_goal}
                            CUSTOMER PROFILE: {profile_text}
                            SEGMENT CHARACTERISTICS: {', '.join(cluster_details['characteristics'][:2])}
                            
                            REQUIREMENTS:
                            1. Use greeting style: {greeting}
                            2. Reference at least 1 specific customer attribute
                            3. Connect to the campaign goal: {campaign_goal}
                            4. Include call-to-action: {cta}
                            5. Keep it 2-3 sentences
                            6. Make it unique and personal
                            7. Use conversational, friendly tone

                            Generate ONLY the marketing message, no quotes or formatting:
                            """
                            
                            try:
                                response = client.chat.completions.create(
                                    model=model_choice,
                                    messages=[
                                        {"role": "system", "content": "You are a creative marketing specialist who writes personalized, engaging messages. Each message should be unique and tailored to the specific customer."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=100,
                                    temperature=0.9,  # Higher temperature for more creativity
                                    presence_penalty=0.6,  # Encourage new content
                                    frequency_penalty=0.6  # Avoid repetition
                                )
                                
                                message = response.choices[0].message.content.strip()
                                # Clean up any quotes
                                message = message.strip('"\'')
                                
                                if debug_mode:
                                    st.write(f"✅ Generated message for customer {customer_idx}: {message[:50]}...")
                                
                                return message
                            
                            except Exception as e:
                                if debug_mode:
                                    st.error(f"API call failed for customer {customer_idx}: {str(e)}")
                                
                                # Enhanced fallback messages with variation
                                fallback_messages = [
                                    f"{greeting}! Based on your profile ({list(customer_data.keys())[0] if customer_data else 'preferences'}), we have something special to help {campaign_goal.lower()}. {cta}",
                                    f"{greeting}! As a valued {cluster_details['name']} member, you'll love our new approach to {campaign_goal.lower()}. {cta}",
                                    f"{greeting}! Your {cluster_details['characteristics'][0].lower()} makes you perfect for our {campaign_goal.lower()} initiative. {cta}",
                                    f"{greeting}! We've designed something special for customers like you to {campaign_goal.lower()}. {cta}"
                                ]
                                return random.choice(fallback_messages)
                        
                        # Initialize all messages with default
                        df['Personalized_Message'] = "Generating personalized message..."
                        df['Message_Type'] = "Template"
                        
                        # Generate unique messages for selected customers
                        content_progress = st.progress(0, text="Generating unique AI messages...")
                        generated_messages = {}
                        api_calls_made = 0
                        total_calls_planned = total_ai_messages
                        
                        if debug_mode:
                            debug_container = st.container()
                        
                        # Create tasks for each cluster
                        for cluster_id in range(num_groups):
                            cluster_data = df[df['Cluster'] == cluster_id].copy()
                            if len(cluster_data) == 0:
                                continue
                            
                            # Sample customers for this cluster
                            n_samples = min(messages_per_group, len(cluster_data))
                            sampled_customers = cluster_data.sample(n=n_samples, random_state=42+cluster_id)  # Different seed for each cluster
                            
                            cluster_details = cluster_info[cluster_id]
                            
                            # Generate messages for this cluster
                            for message_num, (idx, customer_row) in enumerate(sampled_customers.iterrows(), 1):
                                try:
                                    customer_profile = customer_row[all_selected_features].to_dict()
                                    
                                    # Generate unique message
                                    message = generate_unique_message(
                                        idx, 
                                        customer_profile, 
                                        cluster_details, 
                                        campaign_goal,
                                        message_num
                                    )
                                    
                                    generated_messages[idx] = message
                                    df.loc[idx, 'Personalized_Message'] = message
                                    df.loc[idx, 'Message_Type'] = "AI Generated"
                                    
                                    api_calls_made += 1
                                    progress = api_calls_made / total_calls_planned
                                    content_progress.progress(
                                        progress,
                                        text=f"Generated {api_calls_made}/{total_calls_planned} unique messages"
                                    )
                                    
                                    # Small delay to avoid rate limiting
                                    time.sleep(0.1)
                                    
                                except Exception as e:
                                    if debug_mode:
                                        st.error(f"Failed to generate message for customer {idx}: {str(e)}")
                                    api_calls_made += 1
                        
                        # Fill remaining customers with enhanced template messages
                        template_messages = {}
                        for cluster_id in range(num_groups):
                            cluster_details = cluster_info[cluster_id]
                            cluster_mask = (df['Cluster'] == cluster_id) & (df['Message_Type'] == 'Template')
                            
                            if cluster_mask.any():
                                # Create varied template messages
                                templates = [
                                    f"Hello! As a {cluster_details['name']} member, you're invited to {campaign_goal.lower()} with our exclusive offer. Check it out!",
                                    f"Hi there! Your {cluster_details['characteristics'][0].lower()} makes you perfect for our {campaign_goal.lower()} campaign. Learn more!",
                                    f"Greetings! We've designed something special for {cluster_details['name']} customers to {campaign_goal.lower()}. Discover more!",
                                    f"Hello! Join other {cluster_details['name']} members in our mission to {campaign_goal.lower()}. Get started today!"
                                ]
                                
                                # Assign varied templates
                                template_customers = df[cluster_mask].index.tolist()
                                for i, idx in enumerate(template_customers):
                                    template_msg = templates[i % len(templates)]
                                    df.loc[idx, 'Personalized_Message'] = template_msg
                                    template_messages[idx] = template_msg
                        
                        # Add cluster information
                        df['Cluster_Name'] = df['Cluster'].map(lambda x: cluster_info[x]['name'])
                        df['Cluster_Slogan'] = df['Cluster'].map(lambda x: cluster_info[x]['slogan'])
                        
                        main_progress.progress(100, text="Campaign generated successfully! 🎉")
                        time.sleep(1)
                        
                        # Clear progress indicators
                        progress_container.empty()
                    
                    # Results Display
                    st.markdown(f"""
                    <div class="success-animation" style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #00d4aa 0%, #667eea 100%); border-radius: 15px; color: white; margin: 2rem 0; box-shadow: 0 10px 40px rgba(0, 212, 170, 0.4);">
                        <h1>🎉 Campaign Generated Successfully!</h1>
                        <p>Your AI-powered marketing campaign with unique personalized messages</p>
                        <div style="margin-top: 1rem;">
                            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                                🤖 {len(generated_messages)} AI messages • 📝 {len(df) - len(generated_messages)} templates • 💰 ${api_calls_made * 0.002:.3f} cost
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sample of actual AI messages
                    if generated_messages:
                        st.subheader("🤖 Sample Unique AI-Generated Messages")
                        
                        # Display first 6 AI messages to show variety
                        sample_ai_messages = list(generated_messages.items())[:6]
                        
                        cols = st.columns(2)
                        for i, (idx, message) in enumerate(sample_ai_messages):
                            with cols[i % 2]:
                                customer_cluster = df.loc[idx, 'Cluster_Name']
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #00d4aa;">
                                    <strong>🤖 {customer_cluster} (AI)</strong><br>
                                    <em>"{message}"</em><br>
                                    <small style="color: #b3b3b3;">Customer ID: {idx}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show template message samples too
                        if template_messages:
                            st.subheader("📝 Sample Template Messages")
                            sample_template_messages = list(template_messages.items())[:3]
                            
                            for idx, message in sample_template_messages:
                                customer_cluster = df.loc[idx, 'Cluster_Name']
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                                    <strong>📝 {customer_cluster} (Template)</strong><br>
                                    <em>"{message}"</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Campaign Summary
                    st.header("📈 Campaign Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Target Segments", num_groups, delta="AI-Generated")
                    with col2:
                        st.metric("👥 Total Customers", len(df), delta=f"AI: {len(generated_messages)}")
                    with col3:
                        st.metric("🤖 AI Model", model_choice.upper(), delta="GPT-Powered")
                    with col4:
                        cost_estimate = api_calls_made * 0.002
                        st.metric("💰 Actual Cost", f"${cost_estimate:.3f}", delta=f"{api_calls_made} API calls")
                    
                    # Show message uniqueness stats
                    if generated_messages:
                        st.subheader("📊 Message Uniqueness Analysis")
                        
                        # Check for duplicate messages
                        all_messages = list(generated_messages.values())
                        unique_messages = len(set(all_messages))
                        duplicate_rate = (len(all_messages) - unique_messages) / len(all_messages) * 100 if all_messages else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔄 Unique AI Messages", unique_messages, delta=f"{100-duplicate_rate:.1f}% unique")
                        with col2:
                            avg_length = sum(len(msg.split()) for msg in all_messages) / len(all_messages)
                            st.metric("📏 Avg Message Length", f"{avg_length:.1f} words", delta="AI Generated")
                        with col3:
                            st.metric("⚡ Generation Success", f"{len(generated_messages)}/{total_calls_planned}", delta=f"{len(generated_messages)/total_calls_planned*100:.1f}%")
                    
                    # Cluster Visualization
                    st.header("🎨 Customer Segments Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # PCA Scatter Plot with message type
                        fig_scatter = px.scatter(
                            df, x='PCA1', y='PCA2', color='Cluster_Name', symbol='Message_Type',
                            title="Customer Segments with Message Types",
                            hover_data=['Cluster', 'Cluster_Slogan', 'Message_Type'],
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_scatter.update_layout(
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col2:
                        # Message Type Distribution
                        message_type_counts = df['Message_Type'].value_counts()
                        fig_pie = px.pie(
                            values=message_type_counts.values, 
                            names=message_type_counts.index,
                            title="Message Type Distribution",
                            color_discrete_sequence=['#00d4aa', '#667eea']
                        )
                        fig_pie.update_layout(
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Detailed Segment Analysis
                    st.header("🎭 Detailed Segment Analysis")
                    
                    for cluster_id in range(num_groups):
                        cluster_data = df[df['Cluster'] == cluster_id]
                        cluster_details = cluster_info[cluster_id]
                        
                        # Count message types for this cluster
                        ai_messages_in_cluster = len([idx for idx in generated_messages.keys() if df.loc[idx, 'Cluster'] == cluster_id])
                        template_messages_in_cluster = len(cluster_data) - ai_messages_in_cluster
                        
                        with st.expander(f"🎯 {cluster_details['name']} ({len(cluster_data)} customers) - {ai_messages_in_cluster} AI + {template_messages_in_cluster} Template", expanded=True):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**💡 Slogan:** {cluster_details['slogan']}")
                                st.markdown("**🔍 Key Characteristics:**")
                                for char in cluster_details['characteristics']:
                                    st.markdown(f"• {char}")
                                
                                # Show actual messages from this cluster
                                st.markdown("**📝 Sample Messages from this Segment:**")
                                
                                # Get AI messages for this cluster
                                cluster_ai_messages = [
                                    (idx, df.loc[idx, 'Personalized_Message']) 
                                    for idx in generated_messages.keys() 
                                    if df.loc[idx, 'Cluster'] == cluster_id
                                ][:2]
                                
                                # Get template messages for this cluster
                                cluster_template_data = cluster_data[cluster_data['Message_Type'] == 'Template']
                                cluster_template_messages = list(cluster_template_data['Personalized_Message'].head(1))
                                
                                # Display AI messages
                                if cluster_ai_messages:
                                    st.markdown("*🤖 AI Generated:*")
                                    for idx, msg in cluster_ai_messages:
                                        st.markdown(f"• *{msg}*")
                                
                                # Display template messages
                                if cluster_template_messages:
                                    st.markdown("*📝 Template:*")
                                    for msg in cluster_template_messages:
                                        st.markdown(f"• *{msg}*")
                            
                            with col2:
                                # Cluster statistics
                                if selected_numeric:
                                    numeric_cols = [col for col in selected_numeric if col in cluster_data.columns]
                                    if numeric_cols:
                                        fig_box = px.box(
                                            cluster_data, y=numeric_cols[0],
                                            title=f"{numeric_cols[0]} Distribution",
                                            color_discrete_sequence=['#667eea']
                                        )
                                        fig_box.update_layout(
                                            height=300,
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            font_color='white'
                                        )
                                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Export Options
                    st.header("💾 Export Your Campaign")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Full dataset with message types
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📊 Download Full Campaign (CSV)",
                            data=csv_data,
                            file_name=f"marketing_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                    
                    with col2:
                        # AI messages only
                        if generated_messages:
                            ai_messages_df = df.loc[list(generated_messages.keys()), ['Cluster_Name', 'Personalized_Message', 'Cluster_Slogan', 'Message_Type']]
                            ai_csv = ai_messages_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="🤖 Download AI Messages Only (CSV)",
                                data=ai_csv,
                                file_name=f"ai_messages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col3:
                        # Campaign summary with stats
                        summary_data = {
                            'campaign_goal': campaign_goal,
                            'segments': num_groups,
                            'total_customers': len(df),
                            'ai_messages_generated': len(generated_messages),
                            'template_messages': len(df) - len(generated_messages),
                            'api_calls_made': api_calls_made,
                            'estimated_cost': api_calls_made * 0.002,
                            'unique_ai_messages': len(set(generated_messages.values())) if generated_messages else 0,
                            'cluster_info': cluster_info,
                            'generated_at': datetime.now().isoformat()
                        }
                        summary_json = json.dumps(summary_data, indent=2).encode('utf-8')
                        st.download_button(
                            label="📋 Download Campaign Summary (JSON)",
                            data=summary_json,
                            file_name=f"campaign_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # Success animation
                    st.balloons()
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Try checking your file format or data structure.")
        # Add debug info
        with st.expander("🔧 Debug Information"):
            st.write("Error details:", str(e))
            st.write("Error type:", type(e).__name__)
            import traceback
            st.code(traceback.format_exc())

elif uploaded_file is not None and client is None:
    st.warning("⚠️ Please connect your OpenAI API key first to process the data file.")
    
else:
    # Welcome section when no file is uploaded
    st.header("🚀 Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔐 1. Connect API</h3>
            <p>Enter your OpenAI API key in the sidebar to get started. Your key is secure and only used for this session.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 2. Upload Data</h3>
            <p>Upload your customer data in CSV, Excel, or JSON format. CSV separators are automatically detected!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 3. Generate</h3>
            <p>Get truly unique, personalized AI messages with real-time cost tracking and uniqueness analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data section
    st.header("📝 Sample Data Format")
    sample_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'age': [25, 35, 45, 30, 28],
        'income': [50000, 75000, 100000, 60000, 55000],
        'purchase_frequency': [2, 5, 8, 3, 4],
        'preferred_category': ['Electronics', 'Fashion', 'Home', 'Sports', 'Books'],
        'location': ['Urban', 'Suburban', 'Urban', 'Rural', 'Urban']
    })
    
    st.dataframe(sample_data, use_container_width=True)
    st.info("💡 Your data should contain customer attributes like demographics, behavior, preferences, etc. The more detailed your data, the more personalized the AI messages will be!")

# Footer with Dark Theme
st.markdown("""
---
<div style="text-align: center; color: #b3b3b3; padding: 2rem; background: linear-gradient(135deg, #1e1e2e 0%, #262730 100%); border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-top: 2rem;">
    <p> AI Marketing Campaign Generator | Secure API Management | Auto CSV Detection | Unique AI Messages</p>
</div>
""", unsafe_allow_html=True)