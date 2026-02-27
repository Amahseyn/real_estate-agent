# app_faiss.py
import streamlit as st
import openai
import faiss
import pandas as pd
import numpy as np
import os
import pickle
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🏠 Real Estate AI Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS (Enhanced)
# ============================================
st.markdown("""
<style>
    /* Previous CSS styles remain the same... */
    /* Add new animations and improvements */
    
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .search-highlight {
        background: linear-gradient(120deg, #f6e05e 0%, #faf089 100%);
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: 600;
    }
    
    .stats-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .recommendation-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Enhanced FAISS Index Loading
# ============================================
@st.cache_resource
def load_faiss_index(index_path="./real_estate_faiss_index"):
    """Load FAISS index and metadata with enhanced validation"""
    try:
        index_file = os.path.join(index_path, "faiss_index.bin")
        metadata_file = os.path.join(index_path, "metadata.csv")
        config_file = os.path.join(index_path, "config.pkl")
        
        if not all(os.path.exists(f) for f in [index_file, metadata_file, config_file]):
            st.error("❌ FAISS index files not found. Please run create_faiss_index.py first.")
            return None, None, None
        
        # Load FAISS index
        index = faiss.read_index(index_file)
        
        # Load metadata
        metadata = pd.read_csv(metadata_file)
        
        # Load config
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
        
        # Validate index size
        if index.ntotal != len(metadata):
            st.warning(f"⚠️ Index size mismatch: {index.ntotal} vectors vs {len(metadata)} records")
        
        return index, metadata, config
        
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {str(e)}")
        return None, None, None

# ============================================
# Enhanced Search Parameters
# ============================================
class SearchParameterExtractor:
    def __init__(self):
        self.price_patterns = [
            (r'(?:under|below|less than|max|budget of?|up to)\s*\$?\s*([\d,]+)(?:\s*k)?', lambda x: float(x) * 1000 if 'k' in x.lower() else float(x)),
            (r'\$?\s*([\d,]+)(?:\s*k)?\s*(?:to|and|\-)\s*\$?\s*([\d,]+)(?:\s*k)?', lambda x, y: (float(x) * 1000 if 'k' in x.lower() else float(x), 
                                                                                                   float(y) * 1000 if 'k' in y.lower() else float(y))),
            (r'(?:over|above|more than|min)\s*\$?\s*([\d,]+)(?:\s*k)?', lambda x: float(x) * 1000 if 'k' in x.lower() else float(x))
        ]
        
        self.location_patterns = [
            r'in\s+([A-Za-z\s]+?)(?:\s+with|\s+and|\s+under|\s*$|\.)',
            r'near\s+([A-Za-z\s]+?)(?:\s+with|\s+and|\s+under|\s*$|\.)',
            r'around\s+([A-Za-z\s]+?)(?:\s+with|\s+and|\s+under|\s*$|\.)'
        ]
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract all search parameters from text"""
        text_lower = text.lower()
        params = {
            'min_price': None,
            'max_price': None,
            'min_beds': None,
            'max_beds': None,
            'min_baths': None,
            'max_baths': None,
            'min_size': None,
            'max_size': None,
            'location': None,
            'property_type': None,
            'keywords': []
        }
        
        # Extract prices
        for pattern, handler in self.price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) == 1:
                    price = handler(match.group(1))
                    if 'under' in text_lower or 'below' in text_lower or 'max' in text_lower:
                        params['max_price'] = price
                    elif 'over' in text_lower or 'above' in text_lower or 'min' in text_lower:
                        params['min_price'] = price
                elif len(match.groups()) == 2:
                    min_price, max_price = handler(match.group(1), match.group(2))
                    params['min_price'] = min_price
                    params['max_price'] = max_price
        
        # Extract bedrooms
        bed_patterns = [
            (r'(\d+)(?:\+)?\s*(?:bed|br|bd|bedroom)', lambda x: int(x)),
            (r'(?:studio|loft)', lambda: 0),
            (r'(\d+)\s*to\s*(\d+)\s*beds?', lambda x, y: (int(x), int(y)))
        ]
        
        for pattern, handler in bed_patterns:
            if callable(handler):
                match = re.search(pattern, text_lower)
                if match:
                    if len(match.groups()) == 1:
                        params['min_beds'] = handler(match.group(1))
                    elif len(match.groups()) == 2:
                        params['min_beds'], params['max_beds'] = handler(match.group(1), match.group(2))
        
        # Extract bathrooms
        bath_match = re.search(r'(\d+)(?:\+)?\s*(?:bath|ba|bathroom)', text_lower)
        if bath_match:
            params['min_baths'] = int(bath_match.group(1))
        
        # Extract property type
        property_types = ['house', 'home', 'apartment', 'condo', 'townhouse', 'villa', 'mansion', 'cottage']
        for ptype in property_types:
            if ptype in text_lower:
                params['property_type'] = ptype
                break
        
        # Extract location
        for pattern in self.location_patterns:
            match = re.search(pattern, text + ' ', re.IGNORECASE)  # Add space for boundary
            if match:
                location = match.group(1).strip()
                if location and len(location) > 2:
                    params['location'] = location.title()
                    break
        
        # Extract size
        size_match = re.search(r'(\d+)(?:\+)?\s*(?:sq\s*ft|square\s*feet|sqft)', text_lower)
        if size_match:
            params['min_size'] = float(size_match.group(1))
        
        # Extract keywords
        keyword_patterns = ['pool', 'garage', 'garden', 'yard', 'view', 'renovated', 'new', 'updated']
        for kw in keyword_patterns:
            if kw in text_lower:
                params['keywords'].append(kw)
        
        return params

# ============================================
# Enhanced Property Search
# ============================================
class PropertySearcher:
    def __init__(self, index, metadata, config):
        self.index = index
        self.metadata = metadata
        self.config = config
        self.feature_cols = config.get('feature_cols', [])
        
    def create_query_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Create normalized query vector from parameters"""
        # Get normalization parameters
        means = self.config.get('means', {})
        stds = self.config.get('stds', {})
        
        # Default values based on data distribution
        default_values = {
            'price': self.metadata['price'].median() if 'price' in self.metadata else 500000,
            'bed': self.metadata['bed'].median() if 'bed' in self.metadata else 3,
            'bath': self.metadata['bath'].median() if 'bath' in self.metadata else 2,
            'house_size': self.metadata['house_size'].median() if 'house_size' in self.metadata else 2000,
            'acre_lot': self.metadata['acre_lot'].median() if 'acre_lot' in self.metadata else 0.25,
            'price_per_sqft': self.metadata['price_per_sqft'].median() if 'price_per_sqft' in self.metadata else 200,
            'latitude': self.metadata['latitude'].mean() if 'latitude' in self.metadata else 0,
            'longitude': self.metadata['longitude'].mean() if 'longitude' in self.metadata else 0
        }
        
        # Build query values with intelligent defaults
        query_values = {}
        for col in self.feature_cols:
            if col == 'price':
                # Use mid-range if both min and max specified
                if params.get('min_price') and params.get('max_price'):
                    val = (params['min_price'] + params['max_price']) / 2
                elif params.get('max_price'):
                    val = params['max_price'] * 0.7  # Slightly below max
                elif params.get('min_price'):
                    val = params['min_price'] * 1.3  # Slightly above min
                else:
                    val = default_values[col]
            elif col == 'bed':
                val = params.get('min_beds', default_values[col])
            elif col == 'bath':
                val = params.get('min_baths', default_values[col])
            elif col == 'house_size':
                val = params.get('min_size', default_values[col])
            else:
                val = default_values.get(col, 0)
            
            query_values[col] = val
        
        # Normalize
        normalized = []
        for col in self.feature_cols:
            val = query_values.get(col, 0)
            mean = means.get(col, 0)
            std = stds.get(col, 1)
            if std > 0:
                normalized.append((val - mean) / std)
            else:
                normalized.append(0)
        
        return np.array(normalized, dtype='float32').reshape(1, -1)
    
    def search(self, params: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        """Perform enhanced search with multiple strategies"""
        
        # Strategy 1: Vector similarity search
        query_vector = self.create_query_vector(params)
        distances, indices = self.index.search(query_vector, min(top_k * 2, self.index.ntotal))
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                prop = self.metadata.iloc[idx].to_dict()
                
                # Calculate similarity score (0-1)
                distance = distances[0][i]
                similarity = 1.0 / (1.0 + distance)
                
                # Calculate relevance score based on parameter matching
                relevance = self.calculate_relevance(prop, params)
                
                # Combined score
                prop['similarity_score'] = similarity
                prop['relevance_score'] = relevance
                prop['combined_score'] = (similarity * 0.6 + relevance * 0.4)
                prop['distance'] = float(distance)
                
                results.append(prop)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Apply filters
        results = self.apply_filters(results, params)
        
        return results[:top_k]
    
    def calculate_relevance(self, property: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate relevance score based on parameter matching"""
        score = 1.0
        matches = 0
        total = 0
        
        # Price range matching
        if params.get('max_price'):
            total += 1
            if property.get('price', 0) <= params['max_price']:
                matches += 1
                # Bonus for being significantly under budget
                if property['price'] <= params['max_price'] * 0.8:
                    score += 0.2
        
        if params.get('min_price'):
            total += 1
            if property.get('price', 0) >= params['min_price']:
                matches += 1
        
        # Bedroom matching
        if params.get('min_beds'):
            total += 1
            if property.get('bed', 0) >= params['min_beds']:
                matches += 1
            elif property.get('bed', 0) >= params['min_beds'] - 1:
                matches += 0.5  # Partial match
        
        # Bathroom matching
        if params.get('min_baths'):
            total += 1
            if property.get('bath', 0) >= params['min_baths']:
                matches += 1
        
        # Location matching
        if params.get('location'):
            total += 1
            location = str(property.get('city', '')).lower() + ' ' + str(property.get('state', '')).lower()
            if params['location'].lower() in location:
                matches += 1
                score += 0.3
        
        # Keyword matching
        if params.get('keywords'):
            property_text = ' '.join(str(v) for v in property.values()).lower()
            for kw in params['keywords']:
                if kw in property_text:
                    matches += 1
                    score += 0.1
        
        if total > 0:
            score *= (matches / total)
        
        return min(score, 2.0)  # Cap at 2.0
    
    def apply_filters(self, results: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply strict filters to results"""
        filtered = []
        
        for prop in results:
            include = True
            
            # Price filters
            if params.get('max_price') and prop.get('price', 0) > params['max_price']:
                include = False
            if params.get('min_price') and prop.get('price', 0) < params['min_price']:
                include = False
            
            # Bed filters
            if params.get('min_beds') and prop.get('bed', 0) < params['min_beds']:
                include = False
            if params.get('max_beds') and prop.get('bed', 0) > params['max_beds']:
                include = False
            
            # Bath filters
            if params.get('min_baths') and prop.get('bath', 0) < params['min_baths']:
                include = False
            
            # Size filters
            if params.get('min_size') and prop.get('house_size', 0) < params['min_size']:
                include = False
            
            # Property type filter
            if params.get('property_type'):
                prop_type = str(prop.get('property_type', '')).lower()
                if params['property_type'] not in prop_type:
                    include = False
            
            if include:
                filtered.append(prop)
        
        return filtered

# ============================================
# Enhanced Display Functions
# ============================================
def display_enhanced_property_cards(results: List[Dict[str, Any]], recommendations: bool = False):
    """Display properties with enhanced styling and information"""
    if not results:
        return 0
    
    num = len(results)
    
    # Header with stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Properties Found", num, delta=None)
    with col2:
        avg_price = np.mean([p.get('price', 0) for p in results])
        st.metric("Avg Price", f"${avg_price:,.0f}")
    with col3:
        avg_beds = np.mean([p.get('bed', 0) for p in results])
        st.metric("Avg Beds", f"{avg_beds:.1f}")
    with col4:
        avg_baths = np.mean([p.get('bath', 0) for p in results])
        st.metric("Avg Baths", f"{avg_baths:.1f}")
    
    # Display in grid
    for i in range(0, num, 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < num:
                prop = results[idx]
                with cols[j]:
                    display_single_property(prop, idx, recommendations)

def display_single_property(prop: Dict[str, Any], idx: int, recommendations: bool = False):
    """Display a single property card"""
    
    # Extract values with safe defaults
    price = safe_float(prop.get('price', 0))
    beds = safe_int(prop.get('bed', 0))
    baths = safe_int(prop.get('bath', 0))
    house_size = safe_float(prop.get('house_size', 0))
    acre_lot = safe_float(prop.get('acre_lot', 0))
    city = prop.get('city', 'Unknown')
    state = prop.get('state', '')
    status = prop.get('status', 'for_sale')
    street = format_street(prop.get('street', ''))
    similarity = prop.get('similarity_score', 0)
    relevance = prop.get('relevance_score', 0)
    
    # Calculate price per sqft
    price_per_sqft = price / house_size if house_size > 0 else 0
    
    # Status badge class
    status_lower = str(status).lower()
    if "pending" in status_lower:
        status_class, status_text = "status-pending", "Pending"
    elif "sold" in status_lower:
        status_class, status_text = "status-sold", "Sold"
    else:
        status_class, status_text = "status-for-sale", "For Sale"
    
    # Location string
    location = f"{city}, {state}" if state and city != "Unknown" else city
    
    # Build card HTML
    card_html = f"""
    <div class="property-card" style="position: relative;">
        {f'<div class="recommendation-badge">⭐ Top {idx + 1} Pick</div>' if recommendations and idx < 3 else ''}
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span class="property-price">{format_price(price)}</span>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        
        <div style="margin-bottom: 15px;">
            <span class="property-badge badge-bed">🛏️ {beds} Beds</span>
            <span class="property-badge badge-bath">🚿 {baths} Baths</span>
            <span class="property-badge badge-size">📐 {format_size(house_size)}</span>
        </div>
        
        <div class="property-detail">
            <p style="margin: 4px 0;">📍 {location}</p>
            <p style="margin: 4px 0;">🌳 Lot: {format_lot(acre_lot)}</p>
            <p style="margin: 4px 0;">💰 ${price_per_sqft:,.0f}/sqft</p>
            <p style="margin: 4px 0;">📝 {street}</p>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <span style="background: #667eea20; padding: 3px 8px; border-radius: 12px; font-size: 11px;">
                    Match: {similarity:.1%}
                </span>
                <span style="background: #48bb7820; padding: 3px 8px; border-radius: 12px; font-size: 11px;">
                    Relevance: {relevance:.1%}
                </span>
            </div>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def display_price_distribution(results: List[Dict[str, Any]]):
    """Display price distribution chart"""
    if not results:
        return
    
    prices = [p.get('price', 0) for p in results if p.get('price', 0) > 0]
    
    if prices:
        fig = px.histogram(
            prices, 
            nbins=20,
            title="Price Distribution",
            labels={'value': 'Price ($)', 'count': 'Number of Properties'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def display_location_map(results: List[Dict[str, Any]]):
    """Display properties on a map if coordinates available"""
    has_coords = all(p.get('latitude') and p.get('longitude') for p in results[:10])
    
    if has_coords:
        map_data = pd.DataFrame([
            {
                'lat': p['latitude'],
                'lon': p['longitude'],
                'price': p.get('price', 0),
                'address': format_street(p.get('street', ''))
            }
            for p in results[:50] if p.get('latitude') and p.get('longitude')
        ])
        
        if not map_data.empty:
            st.map(map_data)

# ============================================
# Enhanced Intent Detection
# ============================================
class IntentClassifier:
    def __init__(self):
        self.search_patterns = [
            r'\b(find|show|search|look|get|need|want)\b.*\b(house|home|property|apartment|condo)\b',
            r'\b(house|home|property|apartment|condo).*\b(for sale|for rent|to buy)\b',
            r'\b(how many|what are|list|display)\b.*\b(property|properties|listing|listings)\b',
            r'\d+\s*(bed|bath|room|br|ba)',
            r'\$[\d,]+',
            r'\b(under|over|above|below|between)\s*\$?[\d,]+\b',
            r'\b(in|near|around)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*\b'
        ]
        
        self.greeting_patterns = [
            r'^(hi|hello|hey|hola|howdy|sup|yo)$',
            r'^good\s*(morning|afternoon|evening|night)$',
            r'^(what\'?s up|how are you|how do you do)$',
            r'^(thanks|thank you|bye|goodbye|see you)$',
            r'^(who are you|what are you|what can you do|help)$'
        ]
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent with confidence score"""
        text_lower = text.lower().strip()
        
        # Check for greetings
        for pattern in self.greeting_patterns:
            if re.search(pattern, text_lower):
                return 'greeting', 1.0
        
        # Check for property search
        matches = 0
        for pattern in self.search_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            confidence = min(0.5 + matches * 0.1, 1.0)
            return 'property_search', confidence
        elif matches == 1:
            return 'property_search', 0.6
        
        # Check length - very short messages are likely greetings
        if len(text_lower.split()) <= 2:
            return 'greeting', 0.5
        
        return 'unknown', 0.3

# ============================================
# Enhanced GPT Integration
# ============================================
class GPTAssistant:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.setup()
    
    def setup(self):
        try:
            openai.api_key = self.api_key
        except Exception as e:
            st.error(f"❌ Failed to initialize OpenAI: {str(e)}")
    
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 500) -> Optional[str]:
        """Generate response with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    frequency_penalty=0.3,
                    presence_penalty=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    st.error(f"❌ GPT API error: {str(e)}")
                    return None
    
    def generate_search_summary(self, query: str, results: List[Dict[str, Any]], num_found: int) -> str:
        """Generate a smart summary of search results"""
        
        if not results:
            return None
        
        # Extract statistics
        prices = [p.get('price', 0) for p in results if p.get('price', 0) > 0]
        beds = [p.get('bed', 0) for p in results]
        baths = [p.get('bath', 0) for p in results]
        cities = set(p.get('city', 'Unknown') for p in results)
        
        context = {
            'query': query,
            'num_found': num_found,
            'price_range': f"${min(prices):,.0f} - ${max(prices):,.0f}" if prices else "N/A",
            'avg_price': f"${np.mean(prices):,.0f}" if prices else "N/A",
            'bed_range': f"{min(beds)}-{max(beds)} beds",
            'bath_range': f"{min(baths)}-{max(baths)} baths",
            'locations': ', '.join(list(cities)[:5])
        }
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable real estate assistant. "
                    "Provide a concise, helpful summary of the search results. "
                    "Highlight key features and suggest refinements. "
                    "Be enthusiastic but professional. Use emojis sparingly."
                )
            },
            {
                "role": "user",
                "content": f"Search query: '{query}'\nFound {num_found} properties:\n"
                          f"Price range: {context['price_range']}\n"
                          f"Average price: {context['avg_price']}\n"
                          f"Bedrooms: {context['bed_range']}\n"
                          f"Bathrooms: {context['bath_range']}\n"
                          f"Locations: {context['locations']}\n\n"
                          "Provide a brief summary and suggest how to refine the search."
            }
        ]
        
        return self.generate_response(messages, max_tokens=300)
    
    def generate_greeting(self, user_input: str) -> str:
        """Generate appropriate greeting response"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly real estate assistant. "
                    "Respond warmly and professionally. "
                    "Suggest 2-3 example searches they can try. "
                    "Keep it brief but helpful."
                )
            },
            {"role": "user", "content": user_input}
        ]
        
        response = self.generate_response(messages, max_tokens=200)
        
        if not response:
            # Fallback responses
            if 'thank' in user_input.lower():
                return "You're welcome! 😊 Let me know if you need help finding properties. Try asking about houses in specific cities or with certain features!"
            elif 'bye' in user_input.lower():
                return "Goodbye! 👋 Feel free to come back anytime you need help finding your dream home!"
            else:
                return (
                    "👋 Hello! I'm your AI real estate assistant. I can help you find properties based on your preferences.\n\n"
                    "Try asking:\n"
                    "• 'Show me 3 bedroom houses under $500k'\n"
                    "• 'Find homes in Miami with a pool'\n"
                    "• 'Affordable 2 bed apartments near downtown'"
                )
        
        return response

# ============================================
# Safe Value Helpers (Enhanced)
# ============================================
def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        if pd.isna(value) or value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert to int"""
    try:
        if pd.isna(value) or value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def format_price(price):
    """Format price nicely"""
    val = safe_float(price)
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"

def format_size(size):
    """Format size nicely"""
    val = safe_float(size)
    if val >= 10_000:
        return f"{val/1_000:.1f}K sq ft"
    return f"{val:,.0f} sq ft"

def format_lot(lot):
    """Format lot size"""
    val = safe_float(lot)
    if val >= 1:
        return f"{val:.2f} acres"
    elif val > 0:
        sqft = val * 43560
        return f"{sqft:,.0f} sq ft"
    return "N/A"

def format_street(street):
    """Format street address"""
    if not street or pd.isna(street):
        return "Address on request"
    s = str(street).strip()
    # Hide numeric-only values (likely MLS IDs)
    try:
        float(s)
        return "Address on request"
    except ValueError:
        pass
    if len(s) < 3:
        return "Address on request"
    return s

# ============================================
# Main Application
# ============================================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px; margin-bottom: 2rem; color: white;">
        <h1 style="font-size: 48px; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
            🏠 Real Estate AI Assistant
        </h1>
        <p style="font-size: 18px; opacity: 0.9;">
            Powered by FAISS Vector Search + GPT Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load FAISS index
    with st.spinner("📦 Loading FAISS index..."):
        index, metadata, config = load_faiss_index()
    
    if index is None or metadata is None:
        st.error("❌ Failed to load FAISS index. Please ensure the index is created first.")
        st.info("Run: `python create_faiss_index.py --csv data/realtor-data.csv`")
        return
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        # Model selection
        model = st.selectbox(
            "🤖 GPT Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )
        
        st.markdown("---")
        
        # Search filters
        st.markdown("### 🔍 Search Filters")
        
        col1, col2 = st.columns(2)
        with col1:
            min_beds = st.number_input("Min Beds", 0, 10, 0)
        with col2:
            min_baths = st.number_input("Min Baths", 0, 10, 0)
        
        price_range = st.slider(
            "Price Range ($)",
            min_value=0,
            max_value=5_000_000,
            value=(0, 5_000_000),
            step=50000,
            format="$%d"
        )
        
        property_type = st.selectbox(
            "Property Type",
            ["Any", "House", "Apartment", "Condo", "Townhouse", "Villa"],
            index=0
        )
        
        st.markdown("---")
        
        # Results settings
        st.markdown("### 🎯 Results")
        top_k = st.slider("Number of results", 5, 50, 20)
        show_stats = st.checkbox("Show statistics", True)
        show_map = st.checkbox("Show map", False)
        
        st.markdown("---")
        
        # Action buttons
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("❤️ View Favorites", use_container_width=True):
            if st.session_state.favorites:
                st.session_state.show_favorites = True
        
        st.markdown("---")
        
        # Tips
        st.markdown("""
        <div style="background: #f7fafc; padding: 15px; border-radius: 10px;">
            <h4 style="margin-top: 0;">💡 Tips</h4>
            <ul style="margin-bottom: 0; padding-left: 20px;">
                <li>Be specific about location</li>
                <li>Mention bedroom count</li>
                <li>Include price range</li>
                <li>Add features like 'pool'</li>
                <li>Try different cities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize components
    intent_classifier = IntentClassifier()
    property_searcher = PropertySearcher(index, metadata, config)
    search_extractor = SearchParameterExtractor()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("show_cards") and message.get("results"):
                display_enhanced_property_cards(
                    message["results"],
                    recommendations=message.get("recommendations", False)
                )
    
    # Chat input
    if prompt := st.chat_input("Ask about properties..."):
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Classify intent
        intent, confidence = intent_classifier.classify(prompt)
        
        # Handle based on intent
        if intent == 'greeting' and confidence > 0.5:
            # Greeting response
            if api_key:
                gpt = GPTAssistant(api_key, model)
                response = gpt.generate_greeting(prompt)
            else:
                response = "👋 Hello! I'm your real estate assistant. To help you better, please enter your OpenAI API key in the sidebar."
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        elif intent == 'property_search':
            # Perform search
            with st.spinner("🔍 Searching properties..."):
                
                # Extract search parameters
                params = search_extractor.extract(prompt)
                
                # Apply sidebar filters
                if min_beds > 0:
                    params['min_beds'] = min_beds
                if min_baths > 0:
                    params['min_baths'] = min_baths
                if price_range[1] < 5_000_000:
                    params['max_price'] = price_range[1]
                if price_range[0] > 0:
                    params['min_price'] = price_range[0]
                if property_type != "Any":
                    params['property_type'] = property_type.lower()
                
                # Save to history
                st.session_state.search_history.append({
                    'query': prompt,
                    'params': params,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Perform search
                results = property_searcher.search(params, top_k)
                
                if results:
                    # Display results
                    display_enhanced_property_cards(results, recommendations=True)
                    
                    # Show statistics if enabled
                    if show_stats:
                        with st.expander("📊 Statistics & Analysis"):
                            col1, col2 = st.columns(2)
                            with col1:
                                display_price_distribution(results)
                            with col2:
                                st.metric("Average Price per Sqft", 
                                        f"${np.mean([p.get('price', 0)/max(p.get('house_size', 1), 1) for p in results]):,.0f}")
                    
                    # Show map if enabled
                    if show_map:
                        with st.expander("🗺️ Location Map"):
                            display_location_map(results)
                    
                    # Generate GPT summary if API key available
                    if api_key:
                        gpt = GPTAssistant(api_key, model)
                        summary = gpt.generate_search_summary(prompt, results, len(results))
                        
                        if summary:
                            with st.chat_message("assistant"):
                                st.markdown(summary)
                            
                            # Save to session
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": summary,
                                "show_cards": False
                            })
                    
                    # Also save results for display
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Found **{len(results)}** matching properties.",
                        "show_cards": True,
                        "results": results,
                        "recommendations": True
                    })
                    
                else:
                    # No results found
                    no_results_msg = (
                        "😔 No properties matched your search criteria.\n\n"
                        "**Suggestions:**\n"
                        "• Broaden your price range\n"
                        "• Reduce bedroom/bathroom requirements\n"
                        "• Try a different location\n"
                        "• Remove some filters\n\n"
                        "Would you like to try a different search?"
                    )
                    
                    with st.chat_message("assistant"):
                        st.markdown(no_results_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": no_results_msg,
                        "show_cards": False
                    })
        
        else:
            # Unknown intent - handle gracefully
            response = (
                "I'm not sure I understood. I can help you find properties!\n\n"
                "**Try asking:**\n"
                "• 'Show me 3 bedroom houses under $500k'\n"
                "• 'Find homes in Miami with a pool'\n"
                "• 'Affordable 2 bed apartments'"
            )
            
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "show_cards": False
            })

if __name__ == "__main__":
    main()