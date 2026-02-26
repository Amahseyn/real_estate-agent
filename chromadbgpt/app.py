# app.py
import streamlit as st
import openai
from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings
import re

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Real Estate AI Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .stChatMessage {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        animation: fadeIn 0.5s ease-in-out;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    [data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    .property-card {
        background: white;
        border: none;
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .property-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: gradientMove 3s ease infinite;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .property-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }

    .property-price {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        display: inline-block;
    }

    .property-detail {
        color: #4a5568;
        font-size: 16px;
        line-height: 1.6;
    }

    .property-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .badge-bed {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12), rgba(118, 75, 162, 0.12));
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.25);
    }

    .badge-bath {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.12), rgba(56, 161, 105, 0.12));
        color: #38a169;
        border: 1px solid rgba(72, 187, 120, 0.25);
    }

    .badge-size {
        background: linear-gradient(135deg, rgba(246, 173, 85, 0.12), rgba(237, 137, 54, 0.12));
        color: #ed8936;
        border: 1px solid rgba(246, 173, 85, 0.25);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
        padding: 2rem 1rem;
    }

    .sidebar-header {
        font-size: 24px;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }

    .stButton button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        width: 100%;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }

    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 10px 15px;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 5px;
    }

    .metric-label {
        font-size: 14px;
        color: #718096;
        font-weight: 500;
    }

    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-for-sale {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.12), rgba(56, 161, 105, 0.12));
        color: #38a169;
        border: 1px solid rgba(72, 187, 120, 0.25);
    }

    .status-pending {
        background: linear-gradient(135deg, rgba(236, 201, 75, 0.12), rgba(214, 158, 46, 0.12));
        color: #d69e2e;
        border: 1px solid rgba(236, 201, 75, 0.25);
    }

    .status-sold {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.12), rgba(197, 48, 48, 0.12));
        color: #c53030;
        border: 1px solid rgba(245, 101, 101, 0.25);
    }

    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 30px 0;
    }

    .gradient-text {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        font-weight: 700;
    }

    .success-box {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.12), rgba(56, 161, 105, 0.12));
        border-left: 4px solid #38a169;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2d3748;
    }

    .warning-box {
        background: linear-gradient(135deg, rgba(236, 201, 75, 0.12), rgba(214, 158, 46, 0.12));
        border-left: 4px solid #d69e2e;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2d3748;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(66, 153, 225, 0.12), rgba(49, 130, 206, 0.12));
        border-left: 4px solid #3182ce;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2d3748;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .pulse { animation: pulse 2s infinite; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    @media (max-width: 768px) {
        .property-card { margin: 10px 0; }
        .property-price { font-size: 22px; }
        .metric-value { font-size: 24px; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# DATABASE
# ============================================
@st.cache_resource
def init_chroma_db():
    try:
        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        client = chromadb.PersistentClient(
            path="./real_estate_chroma_db",
            settings=settings
        )
        try:
            return client.get_collection("real_estate_properties")
        except Exception:
            return None
    except Exception as e:
        st.error(f"❌ Failed to initialize Chroma DB: {str(e)}")
        return None


# ============================================
# OPENROUTER / AI
# ============================================
def init_openrouter(api_key):
    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenRouter: {str(e)}")
        return False


def get_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if not api_key and "api_key" in st.session_state:
        api_key = st.session_state.api_key
    return api_key


def call_ai(model, messages, max_tokens=300):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Real Estate AI Assistant"
            }
        )
        return response.choices[0].message.content
    except Exception:
        return None


# ============================================
# INTENT DETECTION
# ============================================
def is_property_search(user_input):
    """
    Returns True ONLY if the user is clearly asking about properties.
    Returns False for greetings, casual chat, or vague messages.
    """
    text = user_input.lower().strip()

    # Short messages without property keywords = not a search
    if len(text.split()) <= 2:
        property_short = [
            'house', 'home', 'property', 'properties', 'apartment',
            'condo', 'villa', 'mansion', 'listing', 'listings'
        ]
        if not any(kw in text for kw in property_short):
            return False

    # Explicit greetings
    greeting_patterns = [
        r'^hi[\s!.,?]*$', r'^hello[\s!.,?]*$', r'^hey[\s!.,?]*$',
        r'^hola[\s!.,?]*$', r'^howdy[\s!.,?]*$', r'^sup[\s!.,?]*$',
        r'^yo[\s!.,?]*$', r'^good\s*(morning|afternoon|evening|night)[\s!.,?]*$',
        r'^what\'?s\s*up[\s!.,?]*$', r'^how\s*are\s*you[\s!.,?]*',
        r'^how\s*do\s*you\s*do', r'^thank', r'^thanks',
        r'^bye[\s!.,?]*$', r'^goodbye', r'^see\s*you',
        r'^ok[\s!.,?]*$', r'^okay[\s!.,?]*$', r'^yes[\s!.,?]*$',
        r'^no[\s!.,?]*$', r'^sure[\s!.,?]*$', r'^great[\s!.,?]*$',
        r'^cool[\s!.,?]*$', r'^awesome[\s!.,?]*$', r'^nice[\s!.,?]*$',
        r'^who\s*are\s*you', r'^what\s*are\s*you',
        r'^what\s*can\s*you\s*do', r'^help[\s!.,?]*$',
        r'^tell\s*me\s*about\s*you',
    ]

    for pattern in greeting_patterns:
        if re.search(pattern, text):
            return False

    # Property search indicators
    property_patterns = [
        r'\d+\s*bed', r'\d+\s*bath', r'\d+\s*room', r'\d+\s*br', r'\d+\s*bd',
        r'bedroom', r'bathroom',
        r'\$[\d,]+', r'under\s+\$', r'below\s+\$', r'above\s+\$',
        r'less\s+than\s+\$', r'more\s+than\s+\$',
        r'\bhouse\b', r'\bhome\b', r'\bhomes\b', r'\bhouses\b',
        r'\bproperty\b', r'\bproperties\b',
        r'\bapartment\b', r'\bcondo\b', r'\btownhouse\b', r'\bvilla\b',
        r'\bduplex\b', r'\bmansion\b', r'\bcottage\b', r'\bbungalow\b',
        r'for\s+sale', r'for\s+rent', r'to\s+buy',
        r'looking\s+for', r'find\s+me', r'show\s+me', r'search\s+for',
        r'i\s+want\s+a?\s*(house|home|property|apartment|condo)',
        r'i\s+need\s+a?\s*(house|home|property|apartment|condo)',
        r'sq\s*ft', r'square\s*feet', r'\bacre\b',
        r'\bpool\b', r'\bgarage\b', r'\bgarden\b', r'\byard\b',
        r'\bcheap\b', r'\bexpensive\b', r'\baffordable\b', r'\bluxury\b',
        r'budget', r'price\s*range',
        r'real\s*estate', r'\blisting\b', r'\blistings\b',
    ]

    for pattern in property_patterns:
        if re.search(pattern, text):
            return True

    # If no property keywords found, not a search
    return False


def get_greeting_response(user_input):
    """Fallback greeting if AI call fails"""
    text = user_input.lower().strip()

    if any(w in text for w in ['thank', 'thanks']):
        return (
            "You're welcome! 😊 Let me know if you need help finding properties.\n\n"
            "**Try asking:**\n"
            "- *\"Show me houses under $300k\"*\n"
            "- *\"Find 3 bedroom homes in Miami\"*"
        )
    if any(w in text for w in ['bye', 'goodbye', 'see you']):
        return "Goodbye! 👋 Come back anytime you need help finding your dream home! 🏡"
    if any(w in text for w in ['who are you', 'what are you', 'what can you do']):
        return (
            "I'm your **Real Estate AI Assistant**! 🏠\n\n"
            "I can:\n"
            "- 🔍 Search properties by location, price, size\n"
            "- 🛏️ Filter by bedrooms and bathrooms\n"
            "- 💰 Find homes in your budget\n\n"
            "**Try:** *\"Show me 3 bedroom houses under $500k\"*"
        )
    return (
        "👋 Hello! I'm your Real Estate AI Assistant!\n\n"
        "Tell me what you're looking for and I'll find matching properties.\n\n"
        "**Example searches:**\n"
        "- *\"3 bedroom houses under $300k\"*\n"
        "- *\"Homes in Miami\"*\n"
        "- *\"Affordable 2 bed apartments\"*\n"
        "- *\"Luxury homes with pool\"*"
    )


# ============================================
# SAFE VALUE HELPERS
# ============================================
def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value)) if value is not None else default
    except (ValueError, TypeError):
        return default


def format_price(price):
    val = safe_float(price)
    return f"${val:,.0f}" if val > 0 else "Price N/A"


def format_size(size):
    val = safe_float(size)
    return f"{val:,.0f} sq ft" if val > 0 else "N/A"


def format_lot(lot):
    val = safe_float(lot)
    return f"{val:.2f} acres" if val > 0 else "N/A"


def format_street(street):
    """Clean up street - hide numeric-only values (MLS IDs)"""
    if not street:
        return "Address on request"
    s = str(street).strip()
    try:
        float(s)
        return "Address on request"
    except ValueError:
        pass
    if len(s) < 3:
        return "Address on request"
    return s


# ============================================
# PROPERTY SEARCH WITH RELEVANCE FILTERING
# ============================================
RELEVANCE_THRESHOLD = 1.5  # ChromaDB distance threshold - lower = more relevant

def search_properties(collection, query_text, min_beds=None, max_price=None):
    """
    Search properties. Returns ONLY relevant results by checking
    ChromaDB distance scores against a threshold.
    """
    if not collection:
        return None

    where_conditions = []
    if min_beds and min_beds > 0:
        where_conditions.append({"bed": {"$gte": min_beds}})
    if max_price and max_price > 0:
        where_conditions.append({"price": {"$lte": max_price}})

    where_clause = None
    if len(where_conditions) == 1:
        where_clause = where_conditions[0]
    elif len(where_conditions) > 1:
        where_clause = {"$and": where_conditions}

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=10,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=10,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            st.error(f"🔍 Search failed: {str(e)}")
            return None

    # ---- FILTER BY RELEVANCE ----
    if not results or not results.get('distances') or not results['distances'][0]:
        return None

    filtered = {
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]]
    }

    for idx, distance in enumerate(results['distances'][0]):
        if distance <= RELEVANCE_THRESHOLD:
            filtered['documents'][0].append(results['documents'][0][idx])
            filtered['metadatas'][0].append(results['metadatas'][0][idx])
            filtered['distances'][0].append(distance)

    # If nothing passed the filter, return None
    if not filtered['documents'][0]:
        return None

    return filtered


def validate_results_match_query(results, query_text):
    """
    Extra validation: check if results actually relate to the query.
    For example, if user asks for "Miami" but results are all in "Dallas",
    that's a bad match.
    """
    if not results or not results.get('metadatas') or not results['metadatas'][0]:
        return None

    text = query_text.lower()

    # Extract city names from query
    query_cities = []
    if results['metadatas'][0]:
        all_cities = set()
        for meta in results['metadatas'][0]:
            c = (meta.get('city') or '').strip().lower()
            if c and c != 'unknown':
                all_cities.add(c)

        # Check if user mentioned a specific city
        for city in all_cities:
            if city in text:
                query_cities.append(city)

    # If user asked for a specific city, filter to only that city
    if query_cities:
        filtered = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        for idx, meta in enumerate(results['metadatas'][0]):
            result_city = (meta.get('city') or '').strip().lower()
            if result_city in query_cities:
                filtered['documents'][0].append(results['documents'][0][idx])
                filtered['metadatas'][0].append(meta)
                filtered['distances'][0].append(results['distances'][0][idx])

        if filtered['documents'][0]:
            return filtered
        else:
            return None

    # Extract price constraints from query
    price_match = re.search(r'under\s+\$?([\d,]+)', text)
    if not price_match:
        price_match = re.search(r'below\s+\$?([\d,]+)', text)
    if not price_match:
        price_match = re.search(r'less\s+than\s+\$?([\d,]+)', text)

    if price_match:
        max_p = float(price_match.group(1).replace(',', ''))
        # Handle "300k" style
        if max_p < 1000:
            max_p *= 1000

        filtered = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        for idx, meta in enumerate(results['metadatas'][0]):
            p = safe_float(meta.get('price', 0))
            if p <= max_p:
                filtered['documents'][0].append(results['documents'][0][idx])
                filtered['metadatas'][0].append(meta)
                filtered['distances'][0].append(results['distances'][0][idx])

        if filtered['documents'][0]:
            return filtered
        else:
            return None

    # Extract bedroom requirements
    bed_match = re.search(r'(\d+)\s*(?:bed|br|bd|bedroom)', text)
    if bed_match:
        min_b = int(bed_match.group(1))
        filtered = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        for idx, meta in enumerate(results['metadatas'][0]):
            b = safe_int(meta.get('bed', 0))
            if b >= min_b:
                filtered['documents'][0].append(results['documents'][0][idx])
                filtered['metadatas'][0].append(meta)
                filtered['distances'][0].append(results['distances'][0][idx])

        if filtered['documents'][0]:
            return filtered
        else:
            return None

    return results


# ============================================
# DISPLAY
# ============================================
def display_metric_dashboard(collection):
    try:
        count = collection.count()
        sample = collection.peek(limit=10)
        cities = set()
        total_price = 0
        price_count = 0

        if sample and 'metadatas' in sample:
            for meta in sample['metadatas']:
                c = (meta.get('city') or '').strip()
                if c and c != 'unknown':
                    cities.add(c)
                p = safe_float(meta.get('price', 0))
                if p > 0:
                    total_price += p
                    price_count += 1

        avg_price = total_price / price_count if price_count > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{count}</div>
                <div class="metric-label">Properties</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(cities)}+</div>
                <div class="metric-label">Cities</div>
            </div>
            """, unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:20px;">{format_price(avg_price)}</div>
                <div class="metric-label">Avg Price</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value pulse">✨</div>
                <div class="metric-label">Live</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass


def display_property_cards(results):
    """Display property cards OUTSIDE chat_message"""
    if not results or not results.get('metadatas') or not results['metadatas'][0]:
        return 0

    num = len(results['metadatas'][0])

    st.markdown(f"""
    <div style="text-align:center; margin:20px 0 10px 0;">
        <h3 class="gradient-text">🏠 {num} {'Property' if num == 1 else 'Properties'} Found</h3>
    </div>
    """, unsafe_allow_html=True)

    for i in range(0, num, 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < num:
                meta = results['metadatas'][0][idx]
                with cols[j]:
                    price = safe_float(meta.get('price', 0))
                    beds = safe_int(meta.get('bed', 0))
                    baths = safe_int(meta.get('bath', 0))
                    city = meta.get('city', 'Unknown') or 'Unknown'
                    state = meta.get('state', '') or ''
                    house_size = safe_float(meta.get('house_size', 0))
                    acre_lot = safe_float(meta.get('acre_lot', 0))
                    status = meta.get('status', 'for_sale') or 'for_sale'
                    street = format_street(meta.get('street', ''))

                    if city == "unknown":
                        city = "Unknown"

                    sl = str(status).lower()
                    if "pending" in sl:
                        sc, sd = "status-pending", "Pending"
                    elif "sold" in sl:
                        sc, sd = "status-sold", "Sold"
                    else:
                        sc, sd = "status-for-sale", "For Sale"

                    loc = f"{city}, {state}" if state else city

                    st.markdown(f"""
                    <div class="property-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                            <span class="property-price">{format_price(price)}</span>
                            <span class="status-badge {sc}">{sd}</span>
                        </div>
                        <div style="margin-bottom:15px;">
                            <span class="property-badge badge-bed">🛏️ {beds} Beds</span>
                            <span class="property-badge badge-bath">🚿 {baths} Baths</span>
                            <span class="property-badge badge-size">📐 {format_size(house_size)}</span>
                        </div>
                        <div class="property-detail">
                            <p style="margin:4px 0;">📍 {loc}</p>
                            <p style="margin:4px 0;">🌳 Lot: {format_lot(acre_lot)}</p>
                            <p style="margin:4px 0;">📝 {street}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    return num


def build_context(results, limit=5):
    """Build text summary for AI"""
    if not results or not results.get('metadatas') or not results['metadatas'][0]:
        return ""
    parts = []
    for i, m in enumerate(results['metadatas'][0][:limit]):
        city = m.get('city', 'Unknown') or 'Unknown'
        state = m.get('state', '') or ''
        loc = f"{city}, {state}" if state else city
        parts.append(
            f"#{i+1}: {format_price(m.get('price',0))} | "
            f"{safe_int(m.get('bed',0))}bd/{safe_int(m.get('bath',0))}ba | "
            f"{format_size(m.get('house_size',0))} | {loc}"
        )
    return "\n".join(parts)


# ============================================
# MAIN
# ============================================
def main():
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:white; border-radius:20px;
                margin-bottom:2rem; box-shadow:0 10px 40px rgba(0,0,0,0.1);">
        <h1 class="gradient-text" style="font-size:42px; margin-bottom:10px;">
            🏠 Real Estate AI Assistant
        </h1>
        <p style="color:#718096; font-size:18px; margin:0;">
            Your intelligent property search companion
        </p>
    </div>
    """, unsafe_allow_html=True)

    collection = init_chroma_db()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)

        if collection:
            display_metric_dashboard(collection)

        st.divider()

        default_key = get_api_key() or ""
        api_key = st.text_input(
            "🔑 OpenRouter API Key", type="password",
            value=default_key, placeholder="sk-or-..."
        )
        if api_key:
            st.session_state.api_key = api_key
            st.markdown('<div class="success-box" style="padding:8px;font-size:14px;">✅ API Key saved</div>',
                        unsafe_allow_html=True)

        st.divider()

        model = st.selectbox("🤖 AI Model", [
            "openai/gpt-3.5-turbo", "openai/gpt-4",
            "anthropic/claude-3-haiku", "google/gemini-pro",
            "meta-llama/llama-3-8b-instruct"
        ], index=0)

        st.divider()
        st.markdown("### 🔍 Quick Filters")
        min_beds = st.slider("Min Bedrooms", 0, 10, 0)
        max_price = st.number_input("Max Price ($)", min_value=0, max_value=50000000, value=0, step=50000)

        # Relevance slider
        st.divider()
        st.markdown("### 🎯 Search Precision")
        relevance = st.slider(
            "Relevance Threshold",
            min_value=0.5, max_value=3.0, value=1.5, step=0.1,
            help="Lower = stricter matching. Higher = more results but less relevant."
        )

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("""
        <div class="info-box" style="margin-top:20px; font-size:14px;">
            <strong>💡 Tips:</strong><br>
            • Ask about specific cities<br>
            • Mention bedroom count<br>
            • Include price ranges<br>
            • Describe your dream home
        </div>
        """, unsafe_allow_html=True)

    # ---- CHECKS ----
    if not collection:
        st.markdown("""
        <div style="text-align:center; padding:40px;">
            <div class="warning-box" style="display:inline-block; max-width:600px; padding:25px;">
                <h3 style="margin-top:0;">⚠️ Database Not Found</h3>
                <p>Load your data first:</p>
                <code style="display:block;padding:12px;background:#2d3748;color:#68d391;border-radius:8px;">
                python load_data.py --csv data/realtor-data.csv</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    current_key = get_api_key()
    if not current_key:
        st.markdown("""
        <div style="text-align:center; padding:40px;">
            <div class="warning-box" style="display:inline-block; max-width:500px; padding:25px;">
                <h3 style="margin-top:0;">🔑 API Key Required</h3>
                <p>Enter your OpenRouter API key in the sidebar.<br>
                Get one at <a href="https://openrouter.ai" target="_blank" style="color:#667eea;">openrouter.ai</a></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not init_openrouter(current_key):
        return

    # Update global threshold from slider
    global RELEVANCE_THRESHOLD
    RELEVANCE_THRESHOLD = relevance

    # ---- CHAT STATE ----
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Real Estate AI Assistant.\n\n"
                "I can help you find properties. Try:\n"
                "- *\"Show me 3 bedroom houses under $300k\"*\n"
                "- *\"Find homes in Miami\"*\n"
                "- *\"Affordable 2 bed apartments\"*\n\n"
                "Use **sidebar filters** to narrow results! 🔍"
            )
        }]

    # ---- RENDER HISTORY ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        # Re-render property cards if they were part of this message
        if msg.get("show_cards") and msg.get("results_data"):
            display_property_cards(msg["results_data"])

    # ---- CHAT INPUT ----
    if prompt := st.chat_input("Ask about properties..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ======= GREETING (not a property search) =======
        if not is_property_search(prompt):
            ai_resp = call_ai(model, [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly real estate AI assistant. "
                        "The user is making casual conversation. "
                        "Respond warmly in 1-2 sentences, then suggest "
                        "2-3 example property searches they can try. "
                        "Do NOT list or search any properties."
                    )
                },
                {"role": "user", "content": prompt}
            ], max_tokens=200)

            if not ai_resp:
                ai_resp = get_greeting_response(prompt)

            with st.chat_message("assistant"):
                st.markdown(ai_resp)
            st.session_state.messages.append({"role": "assistant", "content": ai_resp})
            return

        # ======= PROPERTY SEARCH =======
        with st.spinner("🔍 Searching properties..."):
            results = search_properties(
                collection, prompt,
                min_beds=min_beds if min_beds > 0 else None,
                max_price=max_price if max_price > 0 else None
            )

        # Extra validation - filter out results that don't match query
        if results:
            results = validate_results_match_query(results, prompt)

        has_results = (
            results
            and results.get('metadatas')
            and results['metadatas'][0]
            and len(results['metadatas'][0]) > 0
        )

        if has_results:
            # Show cards
            num_found = display_property_cards(results)

            # AI summary
            context = build_context(results)
            ai_resp = call_ai(model, [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly real estate assistant. "
                        "Properties matching the user's search are shown above. "
                        "Give a brief 2-3 sentence summary of what was found. "
                        "Mention price range, locations, sizes. "
                        "Suggest how to refine the search."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Search: '{prompt}'\n"
                        f"Found {num_found} properties:\n{context}\n"
                        f"Summarize briefly."
                    )
                }
            ])

            if not ai_resp:
                ai_resp = (
                    f"🏠 Found **{num_found}** matching "
                    f"{'property' if num_found == 1 else 'properties'}! "
                    f"Check them out above. Want to narrow it down?"
                )

            with st.chat_message("assistant"):
                st.markdown(ai_resp)

            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_resp,
                "show_cards": True,
                "results_data": results
            })

        else:
            # ======= NOTHING FOUND =======
            ai_resp = call_ai(model, [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful real estate assistant. "
                        "No properties matched the user's search. "
                        "Do NOT make up properties. "
                        "Acknowledge nothing was found and suggest "
                        "2-3 specific alternative searches. Be brief."
                    )
                },
                {
                    "role": "user",
                    "content": f"I searched for: '{prompt}' but nothing matched."
                }
            ])

            if not ai_resp:
                ai_resp = (
                    "😔 No properties matched your search.\n\n"
                    "**Try:**\n"
                    "- Broader terms (*\"houses\"* instead of specific features)\n"
                    "- Higher price range or fewer bedrooms\n"
                    "- Different city or location\n"
                    "- Removing sidebar filters\n\n"
                    "I'll keep looking — try again! 🏡"
                )

            with st.chat_message("assistant"):
                st.markdown(ai_resp)

            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_resp
            })


if __name__ == "__main__":
    main()