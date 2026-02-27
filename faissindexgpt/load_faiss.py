# create_faiss_index.py
import faiss
import pandas as pd
import numpy as np
import os
import argparse
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class FAISSIndexBuilder:
    def __init__(self, index_path="./real_estate_faiss_index"):
        self.index_path = index_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = [
            'price', 'bed', 'bath', 'house_size', 'acre_lot',
            'price_per_sqft', 'latitude', 'longitude'
        ]
        
    def clean_value(self, value, default=0.0):
        """Robust value cleaning"""
        if pd.isna(value) or value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def create_derived_features(self, df):
        """Create additional features for better search accuracy"""
        df = df.copy()
        
        # Price per square foot
        df['price_per_sqft'] = df.apply(
            lambda x: self.clean_value(x['price']) / self.clean_value(x['house_size'], 1) 
            if self.clean_value(x['house_size']) > 0 else 0, 
            axis=1
        )
        
        # Total rooms (beds + baths)
        df['total_rooms'] = df['bed'] + df['bath']
        
        # Property score (simple heuristic)
        df['property_score'] = df.apply(
            lambda x: (
                self.clean_value(x['price']) * 
                self.clean_value(x['house_size']) / 1000
            ) if self.clean_value(x['house_size']) > 0 else 0,
            axis=1
        )
        
        return df
    
    def validate_data(self, df):
        """Validate and clean the dataset"""
        print("🔍 Validating data...")
        
        # Check for required columns
        required_cols = ['price', 'bed', 'bath']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_count:
            print(f"✅ Removed {initial_count - len(df)} duplicates")
        
        # Remove invalid prices
        df = df[df['price'] > 1000]  # Minimum realistic price
        df = df[df['price'] < 100_000_000]  # Maximum realistic price
        
        # Remove invalid bedrooms/bathrooms
        df = df[df['bed'] > 0]
        df = df[df['bath'] > 0]
        df = df[df['bed'] < 50]  # Remove outliers
        df = df[df['bath'] < 50]
        
        return df
    
    def load_data(self, csv_path, sample=None, sample_method='head', random_state=42):
        """
        Load and prepare data with enhanced sampling options
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file
        sample : int or None
            Number of rows to sample (None = all rows)
        sample_method : str
            Sampling method: 'head', 'random', or 'stratified'
        random_state : int
            Random seed for reproducible sampling
        """
        print(f"📂 Loading {csv_path}...")
        
        # First, quickly check total rows without loading everything
        try:
            total_rows = sum(1 for _ in open(csv_path)) - 1  # Subtract header
            print(f"📊 Total rows in CSV: {total_rows:,}")
        except:
            total_rows = None
        
        # Read CSV with appropriate settings
        if sample and sample_method == 'head':
            # Simple head sampling (fastest)
            df = pd.read_csv(csv_path, nrows=sample, low_memory=False)
            print(f"📊 Using first {sample} rows (head sampling)")
            
        elif sample and sample_method == 'random':
            # Random sampling (requires scanning file or using skiprows)
            # This is more accurate but slower for large files
            print(f"📊 Using random sampling of {sample} rows...")
            
            if total_rows and total_rows > sample * 2:
                # Efficient random sampling by reading in chunks
                chunks = []
                chunk_size = min(100000, sample * 10)  # Read in chunks
                
                # Calculate skip pattern for random sampling
                np.random.seed(random_state)
                skip_indices = set()
                
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
                    # Randomly select rows from this chunk
                    chunk_indices = np.random.choice(
                        len(chunk), 
                        size=min(len(chunk), sample - len(chunks)), 
                        replace=False
                    )
                    chunks.append(chunk.iloc[chunk_indices])
                    
                    if len(pd.concat(chunks)) >= sample:
                        break
                
                df = pd.concat(chunks).head(sample)
            else:
                # For smaller files, just sample directly
                df = pd.read_csv(csv_path, low_memory=False).sample(
                    n=sample, random_state=random_state
                )
            
        elif sample and sample_method == 'stratified':
            # Stratified sampling based on price quartiles or location
            print(f"📊 Using stratified sampling of {sample} rows...")
            
            # First read a sample to determine strata
            temp_df = pd.read_csv(csv_path, nrows=min(100000, sample * 10), low_memory=False)
            
            # Create strata based on price quartiles
            if 'price' in temp_df.columns:
                temp_df['price_stratum'] = pd.qcut(
                    temp_df['price'].fillna(temp_df['price'].median()), 
                    q=4, 
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],
                    duplicates='drop'
                )
                
                # Calculate samples per stratum
                stratum_counts = temp_df['price_stratum'].value_counts()
                samples_per_stratum = (sample * stratum_counts / len(temp_df)).round().astype(int)
                
                # Read full file and sample by stratum
                full_df = pd.read_csv(csv_path, low_memory=False)
                full_df['price_stratum'] = pd.qcut(
                    full_df['price'].fillna(full_df['price'].median()), 
                    q=4, 
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],
                    duplicates='drop'
                )
                
                dfs = []
                for stratum, count in samples_per_stratum.items():
                    stratum_df = full_df[full_df['price_stratum'] == stratum]
                    if len(stratum_df) > 0:
                        sampled = stratum_df.sample(
                            n=min(count, len(stratum_df)), 
                            random_state=random_state
                        )
                        dfs.append(sampled)
                
                df = pd.concat(dfs)
                df = df.drop('price_stratum', axis=1)
            else:
                # Fallback to random sampling if no price column
                df = pd.read_csv(csv_path, low_memory=False).sample(
                    n=sample, random_state=random_state
                )
        else:
            # Load all data
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"📊 Loading all rows")
        
        print(f"✅ Loaded {len(df):,} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Store sampling info in metadata
        self.sampling_info = {
            'method': sample_method if sample else 'none',
            'sample_size': len(df),
            'total_rows': total_rows,
            'random_state': random_state if sample_method != 'head' else None
        }
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for indexing"""
        print("🔧 Preparing features...")
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Clean each feature column
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.clean_value(x))
            else:
                df[col] = 0.0
                print(f"⚠️ Column {col} not found, using default 0")
        
        # Extract feature matrix
        feature_matrix = df[self.feature_columns].values
        
        # Handle missing values
        feature_matrix = self.imputer.fit_transform(feature_matrix)
        
        # Normalize features
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Clip extreme values (3 standard deviations)
        feature_matrix = np.clip(feature_matrix, -3, 3)
        
        return feature_matrix.astype('float32'), df
    
    def build_index(self, feature_matrix):
        """Build FAISS index with optimizations"""
        print("📦 Building FAISS index...")
        
        dimension = feature_matrix.shape[1]
        
        # Use IVF index for better performance with large datasets
        if len(feature_matrix) > 10000:
            nlist = int(np.sqrt(len(feature_matrix)))  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            
            # Train the index
            print(f"🔄 Training IVF index with {nlist} clusters...")
            index.train(feature_matrix)
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatL2(dimension)
        
        # Add vectors
        index.add(feature_matrix)
        print(f"✅ Added {index.ntotal:,} vectors to index")
        
        return index
    
    def save_index(self, index, metadata_df, config):
        """Save index and metadata"""
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        
        # Save FAISS index
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        faiss.write_index(index, index_file)
        print(f"✅ FAISS index saved to {index_file}")
        
        # Save metadata
        metadata_file = os.path.join(self.index_path, "metadata.csv")
        metadata_df.to_csv(metadata_file, index=False)
        print(f"✅ Metadata saved to {metadata_file}")
        
        # Save configuration
        config_file = os.path.join(self.index_path, "config.pkl")
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        print(f"✅ Config saved to {config_file}")
        
        # Save feature statistics for debugging
        stats_file = os.path.join(self.index_path, "feature_stats.json")
        stats = {
            'feature_columns': self.feature_columns,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else [],
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else [],
            'imputer_statistics': self.imputer.statistics_.tolist() if hasattr(self.imputer, 'statistics_') else [],
            'index_size': index.ntotal,
            'index_type': type(index).__name__,
            'created_at': datetime.now().isoformat(),
            'sampling_info': getattr(self, 'sampling_info', {'method': 'none'})
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Stats saved to {stats_file}")
    
    def run(self, csv_path, sample=None, sample_method='head', random_state=42):
        """Main execution pipeline with enhanced sampling options"""
        try:
            # Load data with sampling
            df = self.load_data(csv_path, sample, sample_method, random_state)
            
            # Validate data
            df = self.validate_data(df)
            
            # Prepare features
            feature_matrix, df = self.prepare_features(df)
            
            # Build index
            index = self.build_index(feature_matrix)
            
            # Prepare config
            config = {
                'feature_cols': self.feature_columns,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'means': dict(zip(self.feature_columns, self.scaler.mean_.tolist())) if hasattr(self.scaler, 'mean_') else {},
                'stds': dict(zip(self.feature_columns, self.scaler.scale_.tolist())) if hasattr(self.scaler, 'scale_') else {},
                'created_at': datetime.now().isoformat(),
                'sampling_info': getattr(self, 'sampling_info', {'method': 'none'})
            }
            
            # Save everything
            self.save_index(index, df, config)
            
            print("\n✨ Index creation completed successfully!")
            print(f"📊 Final statistics:")
            print(f"   - Properties indexed: {len(df):,}")
            print(f"   - Feature dimensions: {len(self.feature_columns)}")
            print(f"   - Index location: {self.index_path}")
            
            if sample:
                print(f"\n📊 Sampling details:")
                print(f"   - Method: {sample_method}")
                print(f"   - Sample size: {len(df):,}")
                if hasattr(self, 'sampling_info') and self.sampling_info.get('total_rows'):
                    total = self.sampling_info['total_rows']
                    pct = (len(df) / total * 100) if total else 0
                    print(f"   - Percentage of total: {pct:.1f}% ({total:,} total)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building index: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='Create FAISS index for real estate data')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--sample', type=int, help='Number of rows to sample')
    parser.add_argument('--sample-method', choices=['head', 'random', 'stratified'], 
                       default='head', help='Sampling method (default: head)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--output', default='./real_estate_faiss_index', 
                       help='Output directory for index files')
    
    args = parser.parse_args()
    
    # Print sampling configuration
    if args.sample:
        print(f"\n🔧 Sampling Configuration:")
        print(f"   - Sample size: {args.sample} rows")
        print(f"   - Sampling method: {args.sample_method}")
        print(f"   - Random seed: {args.random_state}")
        print()
    
    builder = FAISSIndexBuilder(index_path=args.output)
    success = builder.run(
        args.csv, 
        sample=args.sample,
        sample_method=args.sample_method,
        random_state=args.random_state
    )
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()