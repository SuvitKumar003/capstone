import streamlit as st
import requests
import google.generativeai as genai
from textblob import TextBlob
from datetime import datetime
import json
import pickle
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
# Page Configuration
st.set_page_config(
    page_title="FinanceHub Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# API CONFIGURATION
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not NEWS_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("API keys not found")

print("✅ API keys loaded successfully")
# Configure Gemini - Using latest Gemini 2.5 Flash for best performance
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")  # Fast and reliable model

# ============================================
# LOCAL STORAGE FUNCTIONS
# ============================================
STORAGE_DIR = Path("finance_app_data")
STORAGE_DIR.mkdir(exist_ok=True)

def save_blogs(blogs):
    """Save blogs to local storage"""
    with open(STORAGE_DIR / "blogs.pkl", "wb") as f:
        pickle.dump(blogs, f)

def load_blogs():
    """Load blogs from local storage"""
    file_path = STORAGE_DIR / "blogs.pkl"
    if file_path.exists():
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return []

def save_summaries(summaries):
    """Save document summaries to local storage"""
    with open(STORAGE_DIR / "summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)

def load_summaries():
    """Load document summaries from local storage"""
    file_path = STORAGE_DIR / "summaries.pkl"
    if file_path.exists():
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return []

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'blogs' not in st.session_state:
    st.session_state.blogs = load_blogs()
if 'blog_counter' not in st.session_state:
    st.session_state.blog_counter = len(st.session_state.blogs) + 1
if 'summaries' not in st.session_state:
    st.session_state.summaries = load_summaries()
if 'summary_counter' not in st.session_state:
    st.session_state.summary_counter = len(st.session_state.summaries) + 1

# ============================================
# BLOG RATING FUNCTIONS
# ============================================
def rate_with_textblob(content):
    polarity = TextBlob(content).sentiment.polarity
    if polarity <= -0.6:
        return 1
    elif polarity <= -0.2:
        return 2
    elif polarity <= 0.2:
        return 3
    elif polarity <= 0.6:
        return 4
    else:
        return 5

def rate_with_gemini(content):
    prompt = (
        "You are an expert finance content reviewer.\n"
        "Rate the following blog on a scale of 1 to 5 based on:\n"
        "- Relevance to finance topics\n"
        "- Depth, clarity, grammar, coherence\n\n"
        "*Only return a single integer (1 to 5)*.\n\n"
        f"Content:\n{content}"
    )
    try:
        res = model.generate_content(prompt)
        return int(res.text.strip())
    except:
        return None

def rate_blog(content):
    gemini_rating = rate_with_gemini(content)
    blob_rating = rate_with_textblob(content)
    if gemini_rating:
        final = round((gemini_rating * 4 + blob_rating) / 5)
        return max(1, min(5, final))
    return blob_rating

# ============================================
# MARKET ALERTS FUNCTIONS
# ============================================
@st.cache_data(ttl=600)
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()
        
        # Check if API returned an error
        if data.get('status') == 'error':
            st.error(f"News API Error: {data.get('message', 'Unknown error')}")
            return []
        
        articles = data.get('articles', [])
        if not articles:
            return []
            
        return [
            f"{article['title']}. {article['description']}"
            for article in articles
            if article.get('title') and article.get('description')
        ]
    except requests.exceptions.Timeout:
        st.error("⏱️ News API request timed out. Please try again.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Network error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"❌ Error fetching news: {str(e)}")
        return []

@st.cache_data(ttl=600)
def get_stock_alerts(news_list):
    if not news_list:
        return []
    prompt = "\n".join(f"- {headline}" for headline in news_list)
    full_prompt = f"""
You are an expert stock market analyst and financial writer.
Below are recent business news headlines.
Your task:
1. Select only the headlines that are likely to move stock prices — up or down.
2. For each one, identify:
   - The affected *company or sector*
   - Whether the effect is *Positive, **Negative, or **Neutral*
   - A detailed, beginner-friendly explanation of why this matters for investors and how it will impact the stocks. Use emojis where appropriate.
Format each item like this:
- [Headline] — [Company/Sector] — [Impact Direction] — [Detailed Explanation]

Headlines:
{prompt}
"""
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip().split("\n")
    except:
        return []

def parse_alerts(lines):
    alerts = []
    for line in lines:
        try:
            if not line.strip() or "—" not in line:
                continue
            parts = [p.strip(" -*—") for p in line.split("—")]
            if len(parts) >= 4:
                alerts.append({
                    "headline": parts[0],
                    "stock": parts[1],
                    "impact": parts[2],
                    "summary": parts[3]
                })
        except:
            continue
    return alerts

# ============================================
# DOCUMENT SUMMARIZER FUNCTIONS
# ============================================
def summarize_document(content, doc_type="general"):
    """Summarize financial document using Gemini"""
    prompt = f"""
You are an expert financial analyst specializing in document analysis.

Document Type: {doc_type}

Please provide a comprehensive summary of this financial document with the following structure:

1. **Executive Summary** (2-3 sentences)
2. **Key Financial Metrics** (bullet points of important numbers, ratios, or figures)
3. **Main Findings** (detailed insights and observations)
4. **Risk Factors** (if any are mentioned)
5. **Recommendations** (actionable insights for stakeholders)

Document Content:
{content}

Provide a clear, professional summary suitable for executives and investors.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# ============================================
# FINANCE TERMS DICTIONARY
# ============================================
FINANCE_TERMS = {
    "Balance Sheet": "बैलेंस शीट / Bilan / Balance general - A financial statement showing assets, liabilities, and equity",
    "Net Profit": "शुद्ध लाभ / Bénéfice net / Utilidad neta - Revenue minus all expenses and taxes",
    "Assets": "परिसंपत्तियाँ / Actifs / Activos - Resources owned by a business with economic value",
    "Liabilities": "देनदारियाँ / Passifs / Pasivos - Financial obligations or debts owed",
    "Revenue": "राजस्व / Chiffre d'affaires / Ingresos - Total income generated from business operations",
    "Depreciation": "मूल्यह्रास / Amortissement / Depreciación - Decrease in asset value over time",
    "Dividend": "लाभांश / Dividende / Dividendo - Payment made to shareholders from profits",
    "Cash Flow": "नकदी प्रवाह / Flux de trésorerie / Flujo de efectivo - Movement of money in and out of business",
    "ROI": "निवेश प्रतिफल / Retour sur investissement / Retorno de inversión - Return on Investment percentage",
    "IPO": "प्रारंभिक सार्वजनिक निर्गम / Offre publique initiale / Oferta pública inicial - Initial Public Offering",
    "Capital Gains": "पूंजी लाभ / Gains en capital / Ganancias de capital - Profit from selling an asset",
    "Equity": "स्वामित्व पूंजी / Capital-actions / Capital accionario - Ownership value in a company",
    "Debt": "ऋण / Dette / Deuda - Money owed to creditors",
    "Portfolio": "निवेश पोर्टफोलियो / Portefeuille / Portafolio - Collection of investments",
    "Mutual Fund": "म्यूचुअल फंड / Fonds commun / Fondo mutuo - Pooled investment vehicle",
    "Bond": "बांड / Obligation / Bono - Fixed-income debt security",
    "Stock": "शेयर / Action / Acción - Share of ownership in a company",
    "Interest Rate": "ब्याज दर / Taux d'intérêt / Tasa de interés - Cost of borrowing money",
    "Inflation": "मुद्रास्फीति / Inflation / Inflación - Rate of price increase over time",
    "GDP": "सकल घरेलू उत्पाद / PIB / PIB - Gross Domestic Product",
    "Market Cap": "बाजार पूंजीकरण / Capitalisation boursière / Capitalización bursátil - Total market value of shares",
    "Bull Market": "तेजी बाजार / Marché haussier / Mercado alcista - Rising market trend",
    "Bear Market": "मंदी बाजार / Marché baissier / Mercado bajista - Falling market trend",
    "Hedge Fund": "हेज फंड / Fonds spéculatif / Fondo de cobertura - Alternative investment fund",
    "Credit Rating": "ऋण रेटिंग / Notation de crédit / Calificación crediticia - Assessment of creditworthiness"
}

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main Theme Colors */
    :root {
        --primary-color: #1E3A8A;
        --secondary-color: #3B82F6;
        --accent-color: #10B981;
        --background-color: #F8FAFC;
        --text-color: #1E293B;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .feature-card h3 {
        color: #1E3A8A;
        margin-top: 0;
    }
    
    /* Alert Cards */
    .alert-card {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid;
    }
    
    .alert-positive {
        border-left-color: #10B981;
        background: linear-gradient(to right, #F0FDF4 0%, white 100%);
    }
    
    .alert-negative {
        border-left-color: #EF4444;
        background: linear-gradient(to right, #FEF2F2 0%, white 100%);
    }
    
    .alert-neutral {
        border-left-color: #6B7280;
        background: linear-gradient(to right, #F9FAFB 0%, white 100%);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
    
    /* Glossary Card */
    .glossary-result {
        background: #F0F9FF;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .glossary-result h4 {
        color: #1E3A8A;
        margin: 0 0 0.5rem 0;
    }
    
    .glossary-result p {
        color: #475569;
        margin: 0;
    }
    
    /* Info Box */
    .info-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>💰 FinanceHub Pro</h1>
    <p>Your Complete Financial Analysis & Learning Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    page = st.radio(
        "Select a Tool:",
        ["🏠 Home & Glossary", "📄 Document Summarizer", "📊 Market Alerts", "✍️ Blog Rating"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    st.metric("Terms Available", len(FINANCE_TERMS))
    st.metric("Total Blogs", len(st.session_state.blogs))
    st.metric("Summaries Created", len(st.session_state.summaries))
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.85rem;'>
        <p>© 2024 FinanceHub Pro</p>
        <p>Powered by Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE 1: HOME & GLOSSARY
# ============================================
if page == "🏠 Home & Glossary":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 👋 Welcome to FinanceHub Pro")
        st.markdown("""
        Your one-stop platform for financial document analysis, real-time market insights, 
        and comprehensive financial education. All powered by advanced AI technology.
        """)
    
    with col2:
        st.markdown("### 🎯 Features")
        st.markdown("- 📄 AI Document Analysis")
        st.markdown("- 📊 Real-time Market Data")
        st.markdown("- ✍️ Blog Rating System")
        st.markdown("- 🔍 Finance Glossary")
    
    st.markdown("---")
    
    # Finance Term Glossary Search
    st.markdown("### 🔍 Financial Terms Glossary")
    st.markdown("Search from our comprehensive database of financial terms and concepts")
    
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_term = st.text_input(
            "Search for a financial term...",
            placeholder="e.g., Balance Sheet, ROI, IPO, Cash Flow...",
            label_visibility="collapsed"
        )
    
    with search_col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Search Logic
    if search_term or search_button:
        search_term_clean = search_term.strip()
        
        # Exact match
        if search_term_clean in FINANCE_TERMS:
            st.markdown(f"""
            <div class="glossary-result">
                <h4>📖 {search_term_clean}</h4>
                <p>{FINANCE_TERMS[search_term_clean]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Partial match
        else:
            matches = [term for term in FINANCE_TERMS.keys() 
                      if search_term_clean.lower() in term.lower()]
            
            if matches:
                st.markdown(f"### Found {len(matches)} matching term(s):")
                for match in matches[:5]:
                    st.markdown(f"""
                    <div class="glossary-result">
                        <h4>📖 {match}</h4>
                        <p>{FINANCE_TERMS[match]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(matches) > 5:
                    st.info(f"Showing top 5 results. {len(matches) - 5} more terms match your search.")
            
            else:
                st.markdown(f"""
                <div class="info-box">
                    <p><strong>Term not found in our current dictionary.</strong></p>
                    <p>We're constantly updating our glossary. Try searching for a related term!</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Popular Terms Section
    st.markdown("---")
    st.markdown("### 🌟 Most Searched Finance Terms")
    
    popular_terms = ["Balance Sheet", "Cash Flow", "ROI", "IPO", "Dividend", 
                     "Capital Gains", "Market Cap", "Interest Rate"]
    
    cols = st.columns(4)
    for idx, term in enumerate(popular_terms):
        with cols[idx % 4]:
            if st.button(f"📌 {term}", key=f"pop_{term}", use_container_width=True):
                st.markdown(f"""
                <div class="glossary-result">
                    <h4>📖 {term}</h4>
                    <p>{FINANCE_TERMS[term]}</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================
# PAGE 2: DOCUMENT SUMMARIZER
# ============================================
elif page == "📄 Document Summarizer":
    st.markdown("### 📄 AI-Powered Financial Document Summarizer")
    st.markdown("Upload or paste your financial documents for instant AI analysis")
    
    # Tabs for input methods
    tab1, tab2 = st.tabs(["📝 Paste Text", "📋 View History"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            doc_title = st.text_input("Document Title", placeholder="e.g., Q4 Financial Report 2024")
        
        with col2:
            doc_type = st.selectbox(
                "Document Type",
                ["General", "Financial Statement", "Balance Sheet", "Income Statement", 
                 "Cash Flow", "Investment Report", "Audit Report", "Business Plan"]
            )
        
        doc_content = st.text_area(
            "Paste your document content here",
            height=300,
            placeholder="Paste the financial document text you want to summarize..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🤖 Generate Summary", use_container_width=True, type="primary"):
                if not doc_content.strip():
                    st.error("Please paste some content to summarize!")
                elif len(doc_content.strip()) < 100:
                    st.warning("Please provide more content for a meaningful summary (minimum 100 characters)")
                else:
                    with st.spinner("🔍 Analyzing document with AI..."):
                        summary = summarize_document(doc_content, doc_type.lower())
                        
                        # Save to session state and local storage
                        summary_obj = {
                            'id': st.session_state.summary_counter,
                            'title': doc_title or f"Document {st.session_state.summary_counter}",
                            'type': doc_type,
                            'content': doc_content[:500] + "..." if len(doc_content) > 500 else doc_content,
                            'summary': summary,
                            'timestamp': datetime.now()
                        }
                        
                        st.session_state.summaries.append(summary_obj)
                        st.session_state.summary_counter += 1
                        save_summaries(st.session_state.summaries)
                        
                        st.success("✅ Summary generated successfully!")
                        st.markdown("### 📊 Summary")
                        st.markdown(summary)
                        st.balloons()
    
    with tab2:
        st.markdown("### 📜 Summary History")
        
        if not st.session_state.summaries:
            st.info("No summaries yet. Create your first summary!")
        else:
            for summary in reversed(st.session_state.summaries):
                with st.expander(f"📄 {summary['title']} - {summary['type']} ({summary['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                    st.markdown("**Original Content (Preview):**")
                    st.text(summary['content'])
                    st.markdown("---")
                    st.markdown("**AI Summary:**")
                    st.markdown(summary['summary'])

# ============================================
# PAGE 3: MARKET ALERTS
# ============================================
elif page == "📊 Market Alerts":
    st.markdown("### 📊 Real-Time Market Alerts")
    st.markdown("AI-powered analysis of breaking financial news and market impact")
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")
    
    st.markdown("---")
    
    # Fetch and display alerts
    with st.spinner("📡 Fetching latest market news and analyzing impact..."):
        news = fetch_news()
        
        if not news:
            st.error("Unable to fetch news. Please check your internet connection or try again later.")
        else:
            alert_lines = get_stock_alerts(news)
            alerts = parse_alerts(alert_lines)
            
            if not alerts:
                st.warning("No significant market-moving news at the moment.")
            else:
                st.success(f"📰 Found {len(alerts)} market alerts")
                
                for alert in alerts:
                    impact_lower = alert['impact'].lower()
                    
                    if 'positive' in impact_lower:
                        icon = "🟢"
                        card_class = "alert-positive"
                        badge_color = "#10B981"
                    elif 'negative' in impact_lower:
                        icon = "🔴"
                        card_class = "alert-negative"
                        badge_color = "#EF4444"
                    else:
                        icon = "⚪"
                        card_class = "alert-neutral"
                        badge_color = "#6B7280"
                    
                    st.markdown(f"""
                    <div class="alert-card {card_class}">
                        <h3 style="margin-top: 0; color: #1E293B;">{icon} {alert['headline']}</h3>
                        <p style="margin: 0.5rem 0;"><strong>📈 Affected:</strong> {alert['stock']}</p>
                        <p style="margin: 0.5rem 0;">
                            <strong>Impact:</strong> 
                            <span style="background-color: {badge_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem;">
                                {alert['impact']}
                            </span>
                        </p>
                        <p style="margin: 0.75rem 0 0 0; color: #475569; line-height: 1.6;">
                            <strong>💡 Analysis:</strong> {alert['summary']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================
# PAGE 4: BLOG RATING
# ============================================
elif page == "✍️ Blog Rating":
    st.markdown("### ✍️ AI-Powered Blog Rating System")
    st.markdown("Create and rate finance blogs using advanced sentiment analysis and AI")
    
    # Tabs
    tab1, tab2 = st.tabs(["📋 All Blogs", "➕ Create Blog"])
    
    # Tab 1: Blog List
    with tab1:
        st.subheader("📚 Published Blogs")
        
        if not st.session_state.blogs:
            st.info("No blogs yet. Create your first blog in the 'Create Blog' tab!")
        else:
            for blog in reversed(st.session_state.blogs):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"### {blog['title']}")
                        st.markdown(f"*By:* **{blog['user_name']}** | *Tag:* `{blog['tag']}` | *Posted:* {blog['time'].strftime('%Y-%m-%d %H:%M')}")
                    
                    with col2:
                        rating = blog['rating']
                        stars = "⭐" * rating
                        st.markdown(f"### {stars}")
                        st.markdown(f"**{rating}/5**")
                    
                    with st.expander("📖 Read Full Blog"):
                        st.write(blog['content'])
                        st.markdown(f"👍 {blog['likes']} likes | 💬 {blog['comments']} comments")
                    
                    st.divider()
    
    # Tab 2: Create Blog
    with tab2:
        st.subheader("✏️ Write a New Blog")
        
        with st.form("blog_form", clear_on_submit=True):
            user_name = st.text_input("Username*", placeholder="Enter your name")
            title = st.text_input("Blog Title*", placeholder="Enter an engaging title")
            content = st.text_area(
                "Blog Content*", 
                placeholder="Write your finance blog here... Share insights, analysis, or market commentary.",
                height=250
            )
            tag = st.text_input("Tag", placeholder="e.g., Stock Market, Personal Finance, Crypto, Trading")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                submitted = st.form_submit_button("🚀 Publish Blog", use_container_width=True, type="primary")
            
            if submitted:
                if not user_name or not title or not content:
                    st.error("❌ Please fill in all required fields (Username, Title, Content)")
                elif len(content) < 50:
                    st.warning("⚠️ Blog content should be at least 50 characters for proper rating")
                else:
                    # Rate the blog
                    with st.spinner("🤖 Rating your blog with AI..."):
                        rating = rate_blog(content)
                    
                    # Create blog object
                    new_blog = {
                        'id': st.session_state.blog_counter,
                        'user_name': user_name,
                        'title': title,
                        'content': content,
                        'tag': tag or "General Finance",
                        'time': datetime.now(),
                        'likes': 0,
                        'comments': 0,
                        'rating': rating
                    }
                    
                    st.session_state.blogs.append(new_blog)
                    st.session_state.blog_counter += 1
                    save_blogs(st.session_state.blogs)
                    
                    st.success(f"✅ Blog published successfully! AI Rating: {rating}/5 {'⭐' * rating}")
                    st.balloons()
                    st.info("💡 Your blog has been saved and is now visible in the 'All Blogs' tab!")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='background: #F1F5F9; padding: 2rem; border-radius: 10px; text-align: center; margin-top: 3rem; border-top: 3px solid #3B82F6;'>
    <h3>🚀 FinanceHub Pro</h3>
    <p style='color: #64748B;'>All-in-one financial analysis platform powered by Google Gemini AI</p>
    <p style='margin-top: 1rem; color: #64748B; font-size: 0.9rem;'>
        <strong>Features:</strong> Document Summarization | Market Alerts | Blog Rating | Finance Glossary
    </p>
    <p style='color: #94A3B8; font-size: 0.85rem; margin-top: 1rem;'>
        © 2024 FinanceHub Pro | Contact: support@financehubpro.com
    </p>
</div>
""", unsafe_allow_html=True)