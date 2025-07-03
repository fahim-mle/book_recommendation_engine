import pandas as pd
from pathlib import Path

# Define file paths
project_root = Path(__file__).parent.parent
initial_data_path = project_root / 'data' / 'initial_data.csv'
processed_data_path = project_root / 'data' / 'processed_data.csv'
google_data_path = project_root / 'data' / 'google_book_data.csv'
output_path = project_root / 'data' / 'common_books.csv'

# Load data
initial_df = pd.read_csv(initial_data_path)
processed_df = pd.read_csv(processed_data_path)
google_df = pd.read_csv(google_data_path)

# Find common ISBNs
initial_isbns = set(initial_df['ISBN'].astype(str))
processed_isbns = set(processed_df['ISBN'].astype(str))
google_isbns = set(google_df['ISBN'].astype(str))

common_isbns = initial_isbns & processed_isbns & google_isbns

print(f"Found {len(common_isbns)} common ISBNs.")

# Filter each DataFrame to only common ISBNs
initial_common = initial_df[initial_df['ISBN'].astype(str).isin(common_isbns)]
processed_common = processed_df[processed_df['ISBN'].astype(str).isin(common_isbns)]
google_common = google_df[google_df['ISBN'].astype(str).isin(common_isbns)]

# Merge on ISBN (left join to keep all columns)
merged = initial_common.merge(processed_common, on='ISBN', suffixes=('_initial', '_processed'))
merged = merged.merge(google_common, on='ISBN', suffixes=('', '_google'))

# Save to CSV
merged.to_csv(output_path, index=False)
print(f"Saved common books data to {output_path}") 