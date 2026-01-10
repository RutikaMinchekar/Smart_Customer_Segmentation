import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="MarketMind| Premium Insights", 
    page_icon="💎", 
    layout="wide"
)

# ----------------- Modern Global CSS -----------------
def apply_custom_styles():
    # Background images for different states
    if not st.session_state.get("logged_in", False):
        bg_url = "https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&q=80&w=2000" # Soft Gradient Blue
    else:
        bg_url = "https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&q=80&w=2000"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    /* Background Setup */
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.1)), url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

   /* 2. THE TOP BAR FIX: Edge-to-Edge stretching */

    .stTabs [data-baseweb="tab-list"] {{

        position: fixed !important;

        top: 0 !important;

        left: 0 !important;

        width: 100vw !important; /* Forces width to 100% of the viewport */

        height: 120px !important;

        background-color: rgba(255, 255, 255, 0.98) !important;

        display: flex !important;

        justify-content: center !important;

        align-items: center !important;

        gap: 100px !important;

        z-index: 99999 !important;

        border-radius: 0 !important; /* Clean straight edge at the top */

        box-shadow: 0px 10px 30px rgba(0,0,0,0.15) !important;

        padding: 0 !important;

    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre;
        font-weight: 700;
        font-size: 18px;
        color: #1e293b;
        border: none;
    }}

    .stTabs [aria-selected="true"] {{
        color: #6366f1 !important;
        border-bottom: 3px solid #6366f1 !important;
    }}

    /* Glass Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(15px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 40px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
    }}

    /* Buttons */
    .stButton>button {{
        width: 100%;
        border-radius: 15px;
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%) !important;
        color: white !important;
        height: 55px;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
    }}

    /* Hide Default UI */
    [data-testid="stHeader"], [data-testid="stSidebar"] {{display: none;}}

    [data-testid="stWidgetLabel"] p {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ----------------- State Management -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

apply_custom_styles()

# ----------------- MAIN APP -----------------
if not st.session_state.logged_in:
    # Spacious Top Bar for Landing Page
    nav_tabs = st.tabs(["🏠 HOME", "✨ FEATURES", "🛡️ LOGIN", "📞 CONTACT"])

    with nav_tabs[0]:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style='text-align: center;'>
                    <h1 style='font-size: 4rem; font-weight: 800; color: white; text-shadow: 2px 2px 10px rgba(0,0,0,0.3);'>MarketMind</h1>
                    <p style='font-size: 1.5rem; color: white; opacity: 0.9;'>Your Data, Our Strategy, Real Growth.</p>
                </div>
            """, unsafe_allow_html=True)
            # --- FEATURES TAB ---
    with nav_tabs[1]:
        st.markdown("<h2 style='text-align:center; color:white; margin-bottom:50px;'>Cutting-Edge Capabilities</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">🧠</span>
                    <h3>K-Means Architecture</h3>
                    <p style='color: white;'>Utilizing unsupervised machine learning to identify hidden patterns in consumer behavior without manual bias.</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">⚡</span>
                    <h3>Real-Time Prediction</h3>
                    <p style='color: white;'>Instantly classify new customers into high-value segments the moment data is entered into the system.</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">🎯</span>
                    <h3>Precision Targeting</h3>
                    <p style='color: white;'>Receive automated marketing strategies tailored specifically to each cluster's spending power and habits.</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">📊</span>
                    <h3>Advanced Visualization</h3>
                    <p style='color: white;'>Interactive scatter plots and cluster centroids allow you to visualize your market positioning in 2D space.</p>
                </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">🛡️</span>
                    <h3>Secure Data Handling</h3>
                    <p style='color: white;'>Enterprise-grade encryption for all customer datasets, ensuring your competitive insights remain private.</p>
                </div>
            """, unsafe_allow_html=True)
        with col6:
            st.markdown("""
                <div class="feature-card">
                    <span class="feature-icon">💎</span>
                    <h3>Premium Export</h3>
                    <p style='color: white;'>Generate executive-ready PDF reports with segment breakdowns to share with your leadership team.</p>
                </div>
            """, unsafe_allow_html=True)

      # --- CONTACT TAB ---
    with nav_tabs[3]:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        col_info, col_form = st.columns([1, 1.5], gap="large")
        
        with col_info:
            st.markdown("""
                <div style='color: white; padding: 20px;'>
                    <h1 style='font-size: 3.5rem; font-weight: 800; margin-bottom: 10px;'>Get in Touch</h1>
                    <p style='font-size: 1.3rem; opacity: 0.9; line-height: 1.6;'>
                        Ready to transform your customer data into actionable growth? 
                        Our elite support team is available for 24/7 consultation.
                    </p>
                    <br><br>
                    <div style='margin-bottom: 40px; border-left: 4px solid #6366f1; padding-left: 20px;'>
                        <h4 style='margin:0; text-transform: uppercase; letter-spacing: 1px; color: black;'>📞 Contact Number</h4>
                        <p style='opacity: 1; font-size: 1.5rem; font-weight: 600;'>+91 9028270841</p>
                    </div>
                    <div style='margin-bottom: 40px; border-left: 4px solid #6366f1; padding-left: 20px;'>
                        <h4 style='margin:0; text-transform: uppercase; letter-spacing: 1px; color: black;'>✉️ Email Address</h4>
                        <p style='opacity: 1; font-size: 1.5rem; font-weight: 600;'>concierge@persona-ai.com</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_form:
           # st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='color: #1e293b; margin-top:0;'>Send an Inquiry</h3>", unsafe_allow_html=True)
            st.text_input("Full Name", placeholder="Enter your name")
            st.text_input("Business Email", placeholder="email@company.com")
            st.text_area("How can we help you?", placeholder="Describe your business needs...", height=150)
            if st.button("Submit Request"):
                st.balloons()
                st.success("Your inquiry has been received. A specialist will reach out shortly.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with nav_tabs[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            #st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;'>Secure Access</h2>", unsafe_allow_html=True)
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.button("Enter Dashboard"):
                st.session_state.logged_in = True
                st.rerun()
           # st.markdown('</div>', unsafe_allow_html=True)

else:
    # Spacious Top Bar for Dashboard
    dash_tabs = st.tabs(["📊 SEGMENTATION", "🚪 LOGOUT"])

    with dash_tabs[0]:
        
        
        st.markdown("<h2 style='color: white; text-shadow: 1px 1px 5px rgba(0,0,0,0.2);'>Consumer Analysis</h2>", unsafe_allow_html=True)
        
        col_ctrl, col_res = st.columns([1, 1.5], gap="large")
        
        with col_ctrl:
      
            st.markdown("<h4 style='color: white;'>Adjust Parameters</h4>", unsafe_allow_html=True)
            income = st.slider("Annual Income (k)", 10, 200, 75)
            spending = st.slider("Spending Score (1-100)", 1, 100)
            if st.button("Predict Customer"):
                # Simulation Logic
                cluster_id = np.random.randint(0, 5)
                st.session_state.last_cluster = cluster_id
                st.session_state.predicted = True
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("predicted", False):
            with col_res:
                cluster_info = {
                    0: ["#4F46E5", "THE PLATINUM TIER", "High Income & High Spending", "Offer concierge services."],
                    1: ["#10B981", "THE STEADY LOYALIST", "Average Income & Average Spending", "Weekly newsletter target."],
                    2: ["#F59E0B", "THE FRUGAL GIANT", "High Income & Low Spending", "Target with luxury investment pieces."],
                    3: ["#EC4899", "THE TREND SETTER", "Low Income & High Spending", "Target with viral social media ads."],
                    4: ["#64748B", "THE VALUE SEEKER", "Low Income & Low Spending", "Promote essential discounts."]
                }
                info = cluster_info[st.session_state.last_cluster]
                
                st.markdown(f"""
                    <div style="background: {info[0]}; color: white; padding: 35px; border-radius: 25px; box-shadow: 0 15px 35px rgba(0,0,0,0.2);">
                        <p style="text-transform: uppercase; letter-spacing: 2px; font-size: 12px;">Market Persona Found</p>
                        <h1 style="color: white; margin: 0; font-size: 2.5rem;">{info[1]}</h1>
                        <hr style="opacity: 0.3; margin: 20px 0;">
                        <p style="font-size: 1.2rem;"><b>Profile:</b> {info[2]}</p>
                        <p style="font-size: 1.2rem;"><b>Action:</b> {info[3]}</p>
                    </div>
                """, unsafe_allow_html=True)

    with dash_tabs[1]:
    
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
           
            if st.button("Confirm Logout"):
                st.session_state.logged_in = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)