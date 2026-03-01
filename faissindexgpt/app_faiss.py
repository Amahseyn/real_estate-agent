import os
import json
import faiss
import pickle
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import re
import streamlit as st

# ====================================
# Page Configuration
# ====================================
st.set_page_config(
    page_title="🏠 Real Estate Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================
# Custom CSS
# ====================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .property-card {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s;
    }
    .property-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .price-tag {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2E7D32;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .intent-search { background: #E3F2FD; color: #1565C0; }
    .intent-greeting { background: #F3E5F5; color: #7B1FA2; }
    .intent-off-topic { background: #FFF3E0; color: #E65100; }
    .chat-user {
        background: #E3F2FD;
        border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .chat-assistant {
        background: #F5F5F5;
        border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .sidebar-info {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================================
# Load environment and API key
# ====================================
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Allow API key input from sidebar if not in .env
if not API_KEY:
    with st.sidebar:
        API_KEY = st.text_input("🔑 Enter OpenRouter API Key", type="password")
        if not API_KEY:
            st.warning("⚠️ Please enter your API key to continue")
            st.stop()


# ====================================
# Initialize OpenAI client (cached)
# ====================================
@st.cache_resource
def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


client = get_client(API_KEY)

SITE_URL = os.getenv("SITE_URL", "http://localhost")
SITE_NAME = os.getenv("SITE_NAME", "Real Estate Assistant")


# ====================================
# Embeddings utility
# ====================================
def get_embedding(text: str, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(model=model, input=text)
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        st.warning(f"⚠️ Embedding error: {e}")
        return np.zeros(1536, dtype="float32")


# ====================================
# Load FAISS index (cached)
# ====================================
@st.cache_resource
def load_faiss_index(index_path="./real_estate_faiss_index"):
    index_file = os.path.join(index_path, "faiss_index.bin")
    metadata_file = os.path.join(index_path, "metadata.csv")
    config_file = os.path.join(index_path, "config.pkl")

    index = faiss.read_index(index_file)
    metadata = pd.read_csv(metadata_file)

    with open(config_file, "rb") as f:
        config = pickle.load(f)

    return index, metadata, config


# ====================================
# Enhanced Intent Classifier
# ====================================
class SmartIntentClassifier:
    def __init__(self, model="openai/gpt-4o-mini"):
        self.model = model

    def classify(self, text: str) -> dict:
        prompt = f"""
Analyze this query and return a JSON with:
1. intent: one of [greeting, property_search, location_question, pricing_question, comparison_question, other, off_topic]
2. is_real_estate_related: boolean
3. confidence: low/medium/high
4. explanation: brief reason

Query: "{text}"

Return ONLY valid JSON.
"""
        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-OpenRouter-Title": SITE_NAME,
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            return {
                "intent": "other",
                "is_real_estate_related": False,
                "confidence": "low",
                "explanation": f"Could not classify: {e}",
            }


# ====================================
# Enhanced Parameter Extractor
# ====================================
class SmartParameterExtractor:
    def __init__(self, model="openai/gpt-4o-mini"):
        self.model = model
        self.valid_locations = ["austin", "dallas", "houston", "san antonio", "texas"]

    def extract(self, text: str, available_columns):
        prompt = f"""
Extract real estate search parameters as JSON.
Available data columns: {available_columns}

Use these fields with null if missing:
- min_price (number)
- max_price (number)
- min_beds (number)
- min_baths (number)
- min_size (number)
- location (string - city/area)
- property_type (string - house, apartment, condo, etc.)

Query: {text}

Return ONLY valid JSON, no other text.
"""
        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-OpenRouter-Title": SITE_NAME,
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            params = json.loads(content.strip())

            if params.get("location"):
                params["location_valid"] = any(
                    valid_loc in params["location"].lower()
                    for valid_loc in self.valid_locations
                )

            return params
        except Exception as e:
            return {}


# ====================================
# Hybrid Property Searcher
# ====================================
class HybridPropertySearcher:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata
        self.available_cities = set(
            metadata["city"].str.lower().unique() if "city" in metadata.columns else []
        )
        self.available_types = set(
            metadata["property_type"].str.lower().unique()
            if "property_type" in metadata.columns
            else []
        )

    def validate_search_params(self, params):
        issues = []

        if params.get("location") and params.get("location_valid") == False:
            issues.append(
                f"📍 Location '{params['location']}' not found in our database"
            )
            params["location"] = None

        if params.get("property_type") and "property_type" in self.metadata.columns:
            if params["property_type"].lower() not in self.available_types:
                issues.append(
                    f"🏠 Property type '{params['property_type']}' not available"
                )
                params["property_type"] = None

        return params, issues

    def structured_filter(self, df, params):
        try:
            if params.get("min_price") is not None and "price" in df.columns:
                df = df[df["price"] >= params["min_price"]]
            if params.get("max_price") is not None and "price" in df.columns:
                df = df[df["price"] <= params["max_price"]]
            if params.get("min_beds") is not None and "bed" in df.columns:
                df = df[df["bed"] >= params["min_beds"]]
            if params.get("min_baths") is not None and "bath" in df.columns:
                df = df[df["bath"] >= params["min_baths"]]
            if params.get("min_size") is not None and "sqft" in df.columns:
                df = df[df["sqft"] >= params["min_size"]]
            if params.get("location") and "city" in df.columns:
                df = df[
                    df["city"].str.contains(params["location"], case=False, na=False)
                ]
            if params.get("property_type") and "property_type" in df.columns:
                df = df[
                    df["property_type"].str.contains(
                        params["property_type"], case=False, na=False
                    )
                ]
        except Exception as e:
            st.warning(f"⚠️ Filter error: {e}")

        return df

    def semantic_search(self, query, top_k=10):
        try:
            vec = get_embedding(query).reshape(1, -1)
            distances, indices = self.index.search(vec, top_k)
            return indices[0]
        except Exception as e:
            st.warning(f"⚠️ Semantic search error: {e}")
            return []

    def search(self, query, params, top_k=5):
        params, issues = self.validate_search_params(params)
        df = self.structured_filter(self.metadata.copy(), params)

        if df.empty:
            return [], issues

        indices = self.semantic_search(query, top_k * 3)

        results = []
        for idx in indices:
            if idx < len(self.metadata):
                row = self.metadata.iloc[idx]
                if row.name in df.index:
                    results.append(row.to_dict())

        return results[:top_k], issues


# ====================================
# Enhanced GPT Assistant
# ====================================
class GPTAssistant:
    def __init__(self, model="openai/gpt-4o"):
        self.model = model

    def generate_response(self, query, properties, memory, intent_info, issues=None):
        if not intent_info.get("is_real_estate_related", True):
            return self._handle_off_topic(query)

        if not properties and issues:
            return self._handle_no_results(query, issues)

        properties_text = "\n".join(
            [
                f"• Price: ${p.get('price', 'N/A'):,}, Beds: {p.get('bed', 'N/A')}, "
                f"Baths: {p.get('bath', 'N/A')}, Sqft: {p.get('sqft', 'N/A')}, "
                f"City: {p.get('city', 'N/A')}, Type: {p.get('property_type', 'N/A')}"
                for p in properties[:3]
            ]
        )

        if len(properties) > 3:
            properties_text += f"\n... and {len(properties) - 3} more properties"

        messages = []

        for msg in memory[-4:]:
            messages.append(msg)

        messages.append(
            {
                "role": "system",
                "content": "You are a professional real estate assistant. Be helpful, concise, and conversational. "
                "If the user asks about topics outside real estate, politely redirect them to real estate queries.",
            }
        )

        messages.append(
            {
                "role": "user",
                "content": f"User Query: {query}\n\nMatching Properties:\n{properties_text}\n\n"
                f"Please provide a helpful response based on these properties.",
            }
        )

        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-OpenRouter-Title": SITE_NAME,
                },
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Sorry, I encountered an error: {str(e)}"

    def _handle_off_topic(self, query):
        prompt = f"""
The user asked: "{query}"

This is a real estate assistant. Respond politely that you can only help with real estate queries,
and suggest they ask about properties, prices, locations, or home features.
Be friendly and helpful in your tone.
"""
        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-OpenRouter-Title": SITE_NAME,
                },
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150,
            )
            return response.choices[0].message.content
        except:
            return "I'm a real estate assistant, so I can only help with property-related questions. Feel free to ask me about homes, prices, locations, or features you're looking for!"

    def _handle_no_results(self, query, issues):
        prompt = f"""
The user asked: "{query}"

No properties were found. Issues: {', '.join(issues)}

Suggest alternative searches, like:
- Different location (if location was invalid)
- Adjusting price range
- Different property type
- Removing some filters

Be helpful and encouraging.
"""
        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-OpenRouter-Title": SITE_NAME,
                },
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content
        except:
            return "I couldn't find any properties matching your criteria. Try a different location, adjust your price range, or remove some filters."


# ====================================
# Helper: Render property cards
# ====================================
def render_property_cards(properties):
    """Render property results as styled cards"""
    if not properties:
        return

    cols = st.columns(min(len(properties), 3))
    for i, prop in enumerate(properties[:6]):
        with cols[i % 3]:
            price = prop.get("price", 0)
            price_str = f"${price:,.0f}" if price else "N/A"
            city = prop.get("city", "N/A")
            beds = prop.get("bed", "N/A")
            baths = prop.get("bath", "N/A")
            sqft = prop.get("sqft", "N/A")
            ptype = prop.get("property_type", "N/A")

            st.markdown(
                f"""
            <div class="property-card">
                <div class="price-tag">{price_str}</div>
                <p>📍 <strong>{city}</strong></p>
                <p>🛏️ {beds} beds &nbsp;|&nbsp; 🛁 {baths} baths &nbsp;|&nbsp; 📐 {sqft} sqft</p>
                <p>🏠 {ptype}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def get_intent_badge(intent):
    """Return styled intent badge"""
    intent_styles = {
        "property_search": ("🔍 Property Search", "intent-search"),
        "location_question": ("📍 Location", "intent-search"),
        "pricing_question": ("💰 Pricing", "intent-search"),
        "comparison_question": ("⚖️ Comparison", "intent-search"),
        "greeting": ("👋 Greeting", "intent-greeting"),
        "off_topic": ("🚫 Off Topic", "intent-off-topic"),
        "other": ("❓ Other", "intent-off-topic"),
    }
    label, css_class = intent_styles.get(intent, ("❓ Unknown", "intent-off-topic"))
    return f'<span class="intent-badge {css_class}">{label}</span>'


# ====================================
# Initialize Session State
# ====================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ====================================
# Sidebar
# ====================================
with st.sidebar:
    st.markdown("## 🏠 Real Estate Assistant")
    st.markdown("---")

    # Load data
    index_path = st.text_input("📂 FAISS Index Path", value="./real_estate_faiss_index")

    if st.button("🔄 Load / Reload Data", use_container_width=True):
        with st.spinner("Loading FAISS index..."):
            try:
                index, metadata, config = load_faiss_index(index_path)
                st.session_state.index = index
                st.session_state.metadata = metadata
                st.session_state.config = config
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(metadata)} properties")
            except Exception as e:
                st.error(f"❌ Error loading index: {e}")
                st.session_state.data_loaded = False

    # Auto-load on first run
    if not st.session_state.data_loaded:
        try:
            index, metadata, config = load_faiss_index(index_path)
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.config = config
            st.session_state.data_loaded = True
        except:
            pass

    if st.session_state.data_loaded:
        metadata = st.session_state.metadata
        st.markdown("---")
        st.markdown("### 📊 Database Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Properties", f"{len(metadata):,}")
        with col2:
            if "city" in metadata.columns:
                st.metric("Cities", metadata["city"].nunique())

        if "price" in metadata.columns:
            st.markdown("### 💰 Price Range")
            min_p = metadata["price"].min()
            max_p = metadata["price"].max()
            avg_p = metadata["price"].mean()
            st.write(f"Min: ${min_p:,.0f}")
            st.write(f"Max: ${max_p:,.0f}")
            st.write(f"Avg: ${avg_p:,.0f}")

        if "city" in metadata.columns:
            st.markdown("### 📍 Available Cities")
            cities = sorted(metadata["city"].unique())
            for city in cities[:15]:
                count = len(metadata[metadata["city"] == city])
                st.write(f"• {city} ({count})")
            if len(cities) > 15:
                st.write(f"... and {len(cities) - 15} more")

        if "property_type" in metadata.columns:
            st.markdown("### 🏠 Property Types")
            for pt in sorted(metadata["property_type"].unique()):
                count = len(metadata[metadata["property_type"] == pt])
                st.write(f"• {pt} ({count})")

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_memory = []
        st.session_state.search_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
    <div class="sidebar-info">
        <strong>💡 Try asking:</strong><br>
        • Show me 3-bedroom homes under $500K<br>
        • What's available in Austin?<br>
        • Compare houses vs condos<br>
        • Find large homes with 4+ baths
    </div>
    """,
        unsafe_allow_html=True,
    )

# ====================================
# Main Content
# ====================================
st.markdown('<div class="main-header">🏠 Intelligent Real Estate Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Find your dream home with AI-powered search</div>',
    unsafe_allow_html=True,
)

# Check if data is loaded
if not st.session_state.data_loaded:
    st.warning("⚠️ Please load the FAISS index first. Check the sidebar for the index path.")
    st.info(
        "Make sure the following files exist in your index directory:\n"
        "- `faiss_index.bin`\n"
        "- `metadata.csv`\n"
        "- `config.pkl`"
    )
    st.stop()

# Initialize components
metadata = st.session_state.metadata
index = st.session_state.index

classifier = SmartIntentClassifier()
extractor = SmartParameterExtractor()
searcher = HybridPropertySearcher(index, metadata)
assistant = GPTAssistant()

# ====================================
# Chat Display
# ====================================
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏠"):
            st.markdown(msg["content"])

            # Show property cards if attached
            if msg.get("properties"):
                render_property_cards(msg["properties"])

            # Show intent badge if attached
            if msg.get("intent"):
                st.markdown(
                    get_intent_badge(msg["intent"]), unsafe_allow_html=True
                )

# ====================================
# Chat Input
# ====================================
if query := st.chat_input("Ask me about properties... (e.g., 'Find 3-bed homes under $400K in Austin')"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Process query
    with st.chat_message("assistant", avatar="🏠"):
        with st.spinner("🤔 Analyzing your query..."):
            # Step 1: Classify intent
            intent_info = classifier.classify(query)
            intent = intent_info.get("intent", "other")

            # Show intent badge
            st.markdown(get_intent_badge(intent), unsafe_allow_html=True)

            # Step 2: Handle based on intent
            if intent == "greeting":
                response = "Hello! 👋 I'm your real estate assistant. I can help you find properties by location, price, size, and more. What are you looking for today?"
                st.markdown(response)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "intent": intent,
                    }
                )

            elif not intent_info.get("is_real_estate_related", True):
                response = assistant._handle_off_topic(query)
                st.markdown(response)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "intent": intent,
                    }
                )

            elif intent in [
                "property_search",
                "location_question",
                "pricing_question",
                "comparison_question",
            ]:
                # Extract parameters
                params = extractor.extract(query, list(metadata.columns))

                # Show extracted parameters
                if params:
                    active_params = {
                        k: v for k, v in params.items() if v is not None
                    }
                    if active_params:
                        with st.expander("🔍 Search Parameters", expanded=False):
                            st.json(active_params)

                # Search
                results, issues = searcher.search(query, params, top_k=5)

                # Show issues
                if issues:
                    for issue in issues:
                        st.warning(issue)

                if not results:
                    response = assistant._handle_no_results(query, issues)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "intent": intent,
                        }
                    )
                else:
                    st.success(f"📈 Found {len(results)} matching properties")

                    # Render property cards
                    render_property_cards(results)

                    # Generate AI response
                    response = assistant.generate_response(
                        query,
                        results,
                        st.session_state.conversation_memory,
                        intent_info,
                        issues,
                    )
                    st.markdown("---")
                    st.markdown(response)

                    # Save to messages
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "intent": intent,
                            "properties": results[:3],
                        }
                    )

                    # Update conversation memory
                    st.session_state.conversation_memory.append(
                        {"role": "user", "content": query}
                    )
                    st.session_state.conversation_memory.append(
                        {"role": "assistant", "content": response}
                    )

                    # Keep memory manageable
                    if len(st.session_state.conversation_memory) > 10:
                        st.session_state.conversation_memory = (
                            st.session_state.conversation_memory[-10:]
                        )

                    # Track search history
                    st.session_state.search_history.append(
                        {
                            "query": query,
                            "results_count": len(results),
                            "params": params,
                        }
                    )

            else:
                response = (
                    "I'm here to help with real estate! You can ask me about:\n"
                    "- 🔍 Finding specific properties\n"
                    "- 💰 Price ranges in different areas\n"
                    "- 🏠 Home features and amenities\n"
                    "- ⚖️ Comparing different properties"
                )
                st.markdown(response)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "intent": intent,
                    }
                )

# ====================================
# Footer with Quick Actions
# ====================================
st.markdown("---")

st.markdown("### ⚡ Quick Searches")
quick_cols = st.columns(4)

quick_searches = [
    "Show me homes under $300K",
    "4-bedroom houses in Austin",
    "Cheapest condos available",
    "Luxury homes over $1M",
]

for i, search_text in enumerate(quick_searches):
    with quick_cols[i]:
        if st.button(search_text, use_container_width=True, key=f"quick_{i}"):
            st.session_state.quick_query = search_text
            st.rerun()

# Handle quick search (rerun with prefilled query)
if "quick_query" in st.session_state:
    query = st.session_state.pop("quick_query")
    # Add to messages and process
    st.session_state.messages.append({"role": "user", "content": query})

    intent_info = classifier.classify(query)
    params = extractor.extract(query, list(metadata.columns))
    results, issues = searcher.search(query, params, top_k=5)

    if results:
        response = assistant.generate_response(
            query,
            results,
            st.session_state.conversation_memory,
            intent_info,
            issues,
        )
    else:
        response = assistant._handle_no_results(query, issues)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "intent": intent_info.get("intent", "other"),
            "properties": results[:3] if results else None,
        }
    )
    st.rerun()