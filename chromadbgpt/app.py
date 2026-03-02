import os
import json
import faiss
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError
import streamlit as st

# ============================================
# ⚙️ Configuration & CSS
# ============================================
load_dotenv()

st.set_page_config(
    page_title="Professional Real Estate AI",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .property-card {
        background: #ffffff;
        border: 1px solid #e0e6ed;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .price-tag {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2ecc71;
    }
    .intent-badge {
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: inline-block;
    }
    .badge-search   { background: #e1f5fe; color: #0288d1; }
    .badge-chat     { background: #f3e5f5; color: #7b1fa2; }
    .badge-offtopic { background: #fce4ec; color: #c62828; }
    .badge-sort     { background: #e8f5e9; color: #2e7d32; }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 🤖 AI Client
# ============================================
@st.cache_resource
def get_ai_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def clean_json_response(content: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content.strip())


# ============================================
# 🔍 Search & Data Logic
# ============================================
@st.cache_resource
def load_assets(path: str = "./real_estate_faiss_index"):
    """Load FAISS index, metadata CSV, and config pickle."""
    try:
        index    = faiss.read_index(os.path.join(path, "faiss_index.bin"))
        metadata = pd.read_csv(os.path.join(path, "metadata.csv"))
        with open(os.path.join(path, "config.pkl"), "rb") as f:
            config = pickle.load(f)

        # Read the true dimension directly from the loaded FAISS index
        actual_dim = index.d
        return index, metadata, config, actual_dim

    except Exception as e:
        st.error(f"Failed to load assets: {e}")
        return None, None, None, 1536


def dim_to_model(dim: int) -> str:
    """
    Return the OpenAI embedding model whose default output
    matches the FAISS index dimension.
    """
    mapping = {
        1536: "text-embedding-3-small",
        3072: "text-embedding-3-large",
         256: "text-embedding-3-small",
         512: "text-embedding-3-small",
    }
    return mapping.get(dim, "text-embedding-3-large")


class HybridSearcher:
    def __init__(
        self,
        client:   OpenAI,
        index,
        metadata: pd.DataFrame,
        dim:      int,
    ):
        self.client   = client
        self.index    = index
        self.metadata = metadata
        self.dim      = dim
        self.model    = dim_to_model(dim)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Fetch embedding and guarantee the vector length equals self.dim.
          A) exact match  → use as-is
          B) too long     → truncate
          C) too short    → zero-pad
        """
        try:
            kwargs = dict(model=self.model, input=text)
            if self.model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dim

            resp = self.client.embeddings.create(**kwargs)
            vec  = np.array(resp.data[0].embedding, dtype="float32")

        except Exception as e:
            st.warning(f"Embedding error: {e}. Using zero vector.")
            return np.zeros(self.dim, dtype="float32")

        # Dimension safety net
        if vec.shape[0] == self.dim:
            return vec
        elif vec.shape[0] > self.dim:
            return vec[: self.dim]
        else:
            pad = np.zeros(self.dim, dtype="float32")
            pad[: vec.shape[0]] = vec
            return pad

    def _apply_filters(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Apply all structured filters from params to dataframe."""
        if params.get("max_price"):
            df = df[df["price"] <= params["max_price"]]
        if params.get("min_price"):
            df = df[df["price"] >= params["min_price"]]
        if params.get("min_beds"):
            df = df[df["bed"] >= params["min_beds"]]
        if params.get("max_beds"):
            df = df[df["bed"] <= params["max_beds"]]
        if params.get("location"):
            df = df[
                df["city"].str.contains(
                    params["location"], case=False, na=False
                )
            ]
        return df

    def sort_search(self, params: dict, sort_by: str,
                    ascending: bool, top_k: int = 5):
        """
        Pure DataFrame sort — no FAISS involved.
        Used for queries like 'highest price', 'cheapest home', etc.
        Returns (results_list, warnings_list).
        """
        df = self._apply_filters(self.metadata.copy(), params)

        if df.empty:
            return [], ["No properties match those filters."]

        # Map sort_by alias → actual column name
        col_map = {
            "price": "price",
            "beds":  "bed",
            "baths": "bath",
        }
        col = col_map.get(sort_by, "price")

        if col not in df.columns:
            return [], [f"Column '{col}' not found in dataset."]

        df_sorted = df.sort_values(col, ascending=ascending)
        results   = df_sorted.head(top_k).to_dict(orient="records")
        return results, []

    def semantic_search(self, query: str, params: dict, top_k: int = 5):
        """
        FAISS semantic search over filtered subset.
        Returns (results_list, warnings_list).
        """
        df = self._apply_filters(self.metadata.copy(), params)

        if df.empty:
            return [], ["No properties match those specific filters."]

        xq = self.get_embedding(query).reshape(1, -1)

        k = min(top_k * 6, self.index.ntotal)
        if k == 0:
            return [], ["The search index is empty."]

        _, indices = self.index.search(xq, k)

        filtered_idx  = set(df.index.tolist())
        final_results = []
        for idx in indices[0]:
            if idx != -1 and idx in filtered_idx:
                final_results.append(df.loc[idx].to_dict())
            if len(final_results) >= top_k:
                break

        if not final_results:
            return [], ["Semantic search found no results in the filtered set."]

        return final_results, []


# ============================================
# 🧠 Real Estate Agent
# ============================================

_OFF_TOPIC_EXAMPLES = (
    "sports, weather, cooking recipes, medical advice, coding help, "
    "politics, entertainment, travel directions, general trivia"
)

_CLASSIFY_SYSTEM = f"""
You are a classifier for a real estate assistant application.
Your ONLY job is to analyse the user's latest message and return a JSON object.

Rules:
1. "is_real_estate": true  → the message is about buying, renting, selling,
   investing in property, neighbourhoods, mortgages, or home features.
2. "is_real_estate": false → the message is about anything else
   (e.g. {_OFF_TOPIC_EXAMPLES}).

3. "intent" values:
   - "sort"         → user wants listings sorted/ranked by a field.
                      e.g. "highest price", "cheapest", "most bedrooms",
                           "lowest price", "most expensive", "fewest baths"
   - "search"       → user wants property listings by description/location.
   - "conversation" → user asks a real-estate question but NOT for listings.
   - "off_topic"    → not real-estate related at all.

4. "sort_by"   → "price" | "beds" | "baths"  (only when intent = "sort")
5. "ascending" → true = lowest first, false = highest first
                 (only when intent = "sort")
6. Extract numeric/string filters ONLY when clearly stated.

Return ONLY valid JSON, no prose:
{{
  "intent":        "sort" | "search" | "conversation" | "off_topic",
  "is_real_estate": bool,
  "sort_by":       "price" | "beds" | "baths" | null,
  "ascending":     true | false | null,
  "params": {{
    "max_price": int | null,
    "min_price": int | null,
    "min_beds":  int | null,
    "max_beds":  int | null,
    "location":  str | null
  }}
}}

Examples:
- "show me the highest price"      → intent=sort, sort_by=price, ascending=false
- "cheapest homes"                 → intent=sort, sort_by=price, ascending=true
- "most bedrooms"                  → intent=sort, sort_by=beds,  ascending=false
- "homes in Ponce under 150000"    → intent=search, params.location=Ponce, params.max_price=150000
- "what is a cap rate?"            → intent=conversation
- "who won the game last night?"   → intent=off_topic, is_real_estate=false
"""

_RESPONSE_SYSTEM = """
You are a professional real estate consultant with deep market knowledge.

Guidelines:
- Answer ONLY real-estate-related questions.
- If property listings are provided, reference specific details
  (price, beds, city) and summarise clearly.
- For sorted results, explicitly state the sort order used
  (e.g. "Here are the most expensive properties…").
- If no listings match, say so and suggest broadening the search.
- Be concise, warm, and helpful.
- Do NOT answer questions unrelated to real estate; politely redirect instead.
"""

_OFF_TOPIC_REPLY = (
    "I'm your dedicated **real estate assistant** 🏠 and I'm only able to help "
    "with property searches, market questions, mortgages, and related topics.\n\n"
    "Could I help you find a home, explore a neighbourhood, or answer a "
    "real-estate question instead?"
)


class RealEstateAgent:
    def __init__(self, client: OpenAI):
        self.client      = client
        self.model_fast  = "openai/gpt-4o-mini"
        self.model_smart = "openai/gpt-4o"

    # ------------------------------------------------------------------
    # Step 1 – Classify & extract parameters
    # ------------------------------------------------------------------
    def classify_and_extract(self, query: str) -> dict:
        """
        Returns dict with keys:
          intent, is_real_estate, sort_by, ascending, params.
        Falls back to off_topic on any failure.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model_fast,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user",   "content": query},
                ],
                response_format={"type": "json_object"},
                max_tokens=250,
                temperature=0,
            )
            data = clean_json_response(resp.choices[0].message.content)

            # Guarantee all keys exist
            data.setdefault("intent",         "off_topic")
            data.setdefault("is_real_estate",  False)
            data.setdefault("sort_by",         None)
            data.setdefault("ascending",       None)
            data.setdefault("params",          {})

            # Safety: if not real estate, force off_topic
            if not data["is_real_estate"]:
                data["intent"] = "off_topic"

            return data

        except Exception as exc:
            st.warning(f"Classification error: {exc}")
            return {
                "intent":         "off_topic",
                "is_real_estate": False,
                "sort_by":        None,
                "ascending":      None,
                "params":         {},
            }

    # ------------------------------------------------------------------
    # Step 2 – Generate a grounded response
    # ------------------------------------------------------------------
    def generate_response(
        self,
        query:              str,
        context_properties: list,
        history:            list,
        intent:             str = "search",
        sort_by:            str = None,
        ascending:          bool = None,
    ) -> str:
        """
        Build a contextual response using the smart model.
        """
        if context_properties:
            prop_lines = "\n".join([
                f"- ${p.get('price', 0):,} | "
                f"{p.get('bed', '?')} bed / {p.get('bath', '?')} bath | "
                f"{p.get('city', 'Unknown')}"
                for p in context_properties
            ])

            # Give the model a clear hint about sort context
            if intent == "sort" and sort_by:
                direction  = "lowest → highest" if ascending else "highest → lowest"
                sort_label = f"sorted by {sort_by} ({direction})"
                prop_context = (
                    f"Properties {sort_label}:\n{prop_lines}\n\n"
                    f"Please summarise these results clearly, "
                    f"mentioning the sort order."
                )
            else:
                prop_context = f"Relevant listings:\n{prop_lines}"
        else:
            prop_context = (
                "No specific listings were retrieved for this query. "
                "Answer based on general real estate knowledge."
            )

        messages = (
            [{"role": "system", "content": _RESPONSE_SYSTEM}]
            + history
            + [{"role": "user", "content": f"{query}\n\n{prop_context}"}]
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_smart,
                messages=messages,
                temperature=0.7,
                max_tokens=450,
            )
            return resp.choices[0].message.content

        except APIStatusError as e:
            if e.status_code == 402:
                return (
                    "💡 **API Notice**: Your OpenRouter account needs a credit "
                    "top-up. "
                    "[Refill here](https://openrouter.ai/settings/credits)."
                )
            return f"⚠️ API error ({e.status_code}). Please try again shortly."

        except Exception:
            return "⚠️ An unexpected error occurred. Please try again."


# ============================================
# 🖥️ UI Helpers
# ============================================
def render_cards(properties: list):
    """Display up to 6 property cards in a 3-column grid."""
    if not properties:
        return
    cols = st.columns(min(len(properties), 3))
    for i, p in enumerate(properties[:6]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="property-card">
                <div class="price-tag">${p.get('price', 0):,}</div>
                <div style="margin:10px 0;">
                    📍 <strong>{p.get('city', 'N/A')}</strong>
                </div>
                <div style="font-size:0.85rem; color:#555;">
                    🛏️ {p.get('bed', 0)} Beds &nbsp;|&nbsp;
                    🛁 {p.get('bath', 0)} Baths
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_intent_badge(intent: str):
    """Render a coloured intent badge."""
    cfg = {
        "search":       ("badge-search",   "🔍 Property Search"),
        "sort":         ("badge-sort",     "📊 Sorted Results"),
        "conversation": ("badge-chat",     "💬 RE Question"),
        "off_topic":    ("badge-offtopic", "🚫 Off-Topic"),
    }
    css_class, label = cfg.get(intent, ("badge-chat", "💬 Chat"))
    st.markdown(
        f'<span class="intent-badge {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


# ============================================
# 🚀 Main App
# ============================================
def main():
    st.markdown(
        '<div class="main-header">🏠 Real Estate AI Assistant</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Control Panel")
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("API_KEY", ""),
        )
        index_path = st.text_input(
            "Data Path",
            value="./real_estate_faiss_index",
        )
        top_k = st.slider("Max results", min_value=1, max_value=10, value=5)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption(
            "Ask about properties, prices, locations, or real estate topics.\n\n"
            "**Try:** 'Show highest price' or 'Cheapest homes in Ponce'"
        )

    if not api_key:
        st.info("👈 Please enter your OpenRouter API key in the sidebar.")
        return

    # ── Load resources ────────────────────────────────────────────────
    client                       = get_ai_client(api_key)
    index, metadata, config, dim = load_assets(index_path)

    if index is None:
        return

    st.sidebar.info(f"📐 Index dimension: {dim}")

    agent    = RealEstateAgent(client)
    searcher = HybridSearcher(client, index, metadata, dim)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render chat history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("props"):
                render_cards(msg["props"])

    # ── Handle new input ──────────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask me about properties, market trends, or mortgages…"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):

                # Step 1 – Classify ───────────────────────────────────
                analysis  = agent.classify_and_extract(prompt)
                intent    = analysis.get("intent",    "off_topic")
                params    = analysis.get("params",    {})
                sort_by   = analysis.get("sort_by",   None)
                ascending = analysis.get("ascending", False)

                render_intent_badge(intent)

                # Step 2 – Off-topic → instant reply ──────────────────
                if intent == "off_topic":
                    st.markdown(_OFF_TOPIC_REPLY)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": _OFF_TOPIC_REPLY,
                        "props":   None,
                    })
                    st.stop()

                # Step 3 – Retrieve results ───────────────────────────
                results  = []
                warnings = []

                if intent == "sort":
                    # Pure DataFrame sort – no FAISS needed
                    results, warnings = searcher.sort_search(
                        params    = params,
                        sort_by   = sort_by   or "price",
                        ascending = ascending if ascending is not None else False,
                        top_k     = top_k,
                    )

                elif intent == "search":
                    # Semantic FAISS search
                    results, warnings = searcher.semantic_search(
                        prompt, params, top_k=top_k
                    )

                # Show any filter warnings
                for w in warnings:
                    st.markdown(
                        f'<div class="warning-box">⚠️ {w}</div>',
                        unsafe_allow_html=True,
                    )

                # Step 4 – Build history (exclude current user turn) ──
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ][-6:]

                # Step 5 – Generate response ──────────────────────────
                response = agent.generate_response(
                    query              = prompt,
                    context_properties = results,
                    history            = history,
                    intent             = intent,
                    sort_by            = sort_by,
                    ascending          = ascending,
                )

                st.markdown(response)
                if results:
                    render_cards(results)

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": response,
                    "props":   results or None,
                })


if __name__ == "__main__":
    main()