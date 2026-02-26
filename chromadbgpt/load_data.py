# load_data.py
import chromadb
from chromadb.config import Settings
import pandas as pd
import os
import argparse

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

def clean_value(value):
    """Clean values for ChromaDB"""
    if pd.isna(value) or value is None:
        return "unknown"
    
    if isinstance(value, (int, float)):
        return value
    
    str_val = str(value).strip()
    if str_val in ['nan', 'None', '', 'null']:
        return "unknown"
    return str_val

def load_data(csv_path, db_path="./real_estate_chroma_db", sample=None):
    """Load data into ChromaDB with telemetry disabled"""
    
    print(f"📂 Loading {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    if sample:
        df = df.head(sample)
    print(f"✅ Loaded {len(df)} rows")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Remove existing DB
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print("🗑️ Removed old database")
    
    # Create new DB with telemetry disabled
    settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
    
    client = chromadb.PersistentClient(
        path=db_path,
        settings=settings
    )
    
    # Create collection
    collection = client.create_collection(
        name="real_estate_properties"
    )
    print("✅ Created new collection")
    
    # Process data
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        try:
            # Create document text for search
            doc_parts = []
            
            # Add key info to document
            city = clean_value(row.get('city'))
            state = clean_value(row.get('state'))
            price = row.get('price')
            bed = row.get('bed')
            bath = row.get('bath')
            
            if city != "unknown":
                doc_parts.append(f"Property in {city}")
            if state != "unknown":
                doc_parts.append(state)
            if pd.notna(price) and price > 0:
                doc_parts.append(f"${float(price):,.0f}")
            if pd.notna(bed):
                doc_parts.append(f"{int(float(bed))} bedroom")
            if pd.notna(bath):
                doc_parts.append(f"{int(float(bath))} bathroom")
            
            document = " ".join(doc_parts) if doc_parts else "Property listing"
            documents.append(document)
            
            # Create metadata
            metadata = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    # Use appropriate defaults
                    if col in ['price', 'acre_lot', 'house_size']:
                        metadata[col] = 0.0
                    elif col in ['bed', 'bath']:
                        metadata[col] = 0
                    else:
                        metadata[col] = "unknown"
                else:
                    if col in ['price', 'acre_lot', 'house_size']:
                        metadata[col] = float(val)
                    elif col in ['bed', 'bath']:
                        metadata[col] = int(float(val))
                    else:
                        metadata[col] = str(val)
            
            metadatas.append(metadata)
            ids.append(f"prop_{idx:06d}")
            
            # Progress
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
            
        except Exception as e:
            print(f"⚠️ Error row {idx}: {e}")
    
    # Add to ChromaDB
    print(f"📦 Adding {len(documents)} documents...")
    
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        try:
            collection.add(
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            print(f"  ✅ Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        except Exception as e:
            print(f"  ❌ Batch failed: {e}")
    
    final_count = collection.count()
    print(f"✅ Successfully added {final_count} properties")
    
    # Show sample
    if final_count > 0:
        sample = collection.peek()
        print(f"\n📋 Sample property:")
        print(f"  Document: {sample['documents'][0][:100]}...")
        print(f"  Metadata: {sample['metadatas'][0]}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--sample", type=int, help="Load only sample rows")
    
    args = parser.parse_args()
    load_data(args.csv, sample=args.sample)