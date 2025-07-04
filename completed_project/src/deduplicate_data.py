"""
Deduplicate engineered data to improve model performance
"""
import pandas as pd
import os
import sys

# Add the src directory to the path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def deduplicate_data():
    """
    Deduplicate data based on ISBN
    """
    print("🔍 Starting data deduplication...")
    
    # Get file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    # Input file: engineered_data.csv
    input_path = os.path.join(data_dir, 'engineered_data.csv')
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False
    
    # Load engineered data
    print(f"📖 Loading engineered data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Check duplicates
    print(f"Before: {len(df)} rows, {df['ISBN'].nunique()} unique ISBNs")
    
    # Keep first occurrence of each ISBN
    df_unique = df.drop_duplicates(subset=['ISBN'], keep='first')
    
    print(f"After: {len(df_unique)} rows, {df_unique['ISBN'].nunique()} unique ISBNs")
    print(f"Removed {len(df) - len(df_unique)} duplicate entries")
    
    # Output file: engineered_data_unique.csv
    output_path = os.path.join(data_dir, 'engineered_data_unique.csv')
    
    # Save deduplicated data
    print(f"💾 Saving deduplicated data to: {output_path}")
    df_unique.to_csv(output_path, index=False)
    
    print("✅ Deduplication complete!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("📚 ENGINEERED DATA DEDUPLICATION")
    print("=" * 60)
    
    deduplicate_data()
    
    print("\n" + "=" * 60) 