# Real Estate AI Assistant 🏠

An intelligent real estate consultation application powered by LLMs (via OpenRouter) and vector search (FAISS/ChromaDB). This project provides a conversational interface for searching property listings, analyzing market trends, and receiving professional real estate advice.

## ✨ Features

- **Hybrid Search**: Combines semantic vector search with structured filtering (price, bedrooms, bathrooms, location).
- **Intent Classification**: Automatically detects if a user is searching for properties, asking a general real estate question, or just greeting the assistant.
- **Intelligent Parameter Extraction**: Extracts specific search criteria directly from natural language queries.
- **Interactive UI**: Built with Streamlit, featuring styled property cards, intent badges, and a professional chat interface.
- **Multiple Implementations**:
    - **Standard (FAISS)**: Advanced indexing with derived features (price per sqft, property scores) and stratified sampling.
    - **ChromaDB**: Alternative vector database implementation.
    - **Simple Chat**: A lightweight template for direct LLM interaction.

## 🏗️ Project Structure

```text
real_estate-agent/
├── app.py                  # Simple OpenRouter Chat (Template)
├── faissindexgpt/          # Main FAISS-based Application
│   ├── app_faiss.py        # Streamlit app with advanced metrics
│   └── load_faiss.py       # Data processing and FAISS index creation
├── chromadbgpt/            # ChromaDB & FAISS Hybrid Application
│   ├── app.py              # Streamlit app with hybrid searcher
│   └── load_data.py        # ChromaDB data loading script
├── data/                   # Raw data files
│   └── realtor-data.csv    # Sample real estate dataset
├── real_estate_faiss_index/# Generated FAISS index & metadata
├── real_estate_chroma_db/  # Generated ChromaDB database
├── tests/                  # Unit and integration tests
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- [OpenRouter API Key](https://openrouter.ai/keys)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd real_estate-agent
   ```

2. **Set up environment**:
   Create a `.env` file in the root directory:
   ```env
   API_KEY=your_openrouter_api_key_here
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 📊 Data Preparation

Before running the apps, you need to initialize the vector indices.

**For FAISS (Recommended):**
```bash
python faissindexgpt/load_faiss.py --csv data/realtor-data.csv --sample 1000 --sample-method random
```

**For ChromaDB:**
```bash
python chromadbgpt/load_data.py --csv data/realtor-data.csv --sample 1000
```

## 💻 Usage

Run the Streamlit applications using the following commands:

**Advanced FAISS Assistant (Highest Performance):**
```bash
streamlit run faissindexgpt/app_faiss.py
```

**Standard Hybrid Assistant:**
```bash
streamlit run chromadbgpt/app.py
```

**Simple Chat Template:**
```bash
streamlit run app.py
```

## 🔍 Examples to Try

- "Show me 3-bedroom homes in Austin under $500,000"
- "What are the most expensive properties in Dallas?"
- "What is a mortgage pre-approval?"
- "Compare the benefits of a condo vs a single-family home."

## 🛠️ Tech Stack

- **Language**: Python 3.12
- **UI Framework**: Streamlit
- **LLM API**: OpenRouter (GPT-4o, GPT-4o-mini, etc.)
- **Vector Search**: FAISS, ChromaDB
- **Data Handling**: Pandas, NumPy, Scikit-learn

## 📄 License

This project is licensed under the MIT License.