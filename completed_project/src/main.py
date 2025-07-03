"""
Main Module - Entry point for the Book Recommendation Engine
"""
import pandas as pd
import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection import APIClient
from src.data_processing import TextProcessor
from src.recommender import ContentBasedRecommender
from src.evaluation import ModelEvaluator

def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def save_data(df, file_path):
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
    """
    print(f"Saving data to {file_path}")
    df.to_csv(file_path, index=False)

def collect_data(api_key, input_file, output_file):
    """
    Collect data from Trove API
    
    Args:
        api_key: Trove API key
        input_file: Path to input CSV file with ISBNs
        output_file: Path to save augmented data
    """
    print("Starting data collection...")
    
    # Load initial data
    initial_data = load_data(input_file)
    
    # Initialize API client
    api_client = APIClient(api_key=api_key)
    
    # Augment data with API
    print("Augmenting data with Trove API...")
    augmented_data = api_client.augment_dataset(initial_data)
    
    # Save augmented data
    save_data(augmented_data, output_file)
    
    print(f"Data collection complete. Augmented data saved to {output_file}")
    print(f"Shape of augmented data: {augmented_data.shape}")

def process_data(input_file, output_file):
    """
    Process and prepare data for the recommendation engine
    
    Args:
        input_file: Path to input CSV file with augmented data
        output_file: Path to save processed data
    """
    print("Starting data processing...")
    
    # Load augmented data
    augmented_data = load_data(input_file)
    
    # Initialize text processor
    processor = TextProcessor()
    
    # Process data
    print("Processing data...")
    processed_data = processor.preprocess(augmented_data)
    
    # Save processed data
    save_data(processed_data, output_file)
    
    print(f"Data processing complete. Processed data saved to {output_file}")
    print(f"Shape of processed data: {processed_data.shape}")

def train_recommender(input_file, output_file=None):
    """
    Train the recommendation engine
    
    Args:
        input_file: Path to input CSV file with processed data
        output_file: Optional path to save model (not implemented yet)
    """
    print("Training recommendation engine...")
    
    # Load processed data
    processed_data = load_data(input_file)
    processed_data = processed_data.reset_index(drop=True)  # Ensure indices are consecutive
    
    # Initialize recommender
    recommender = ContentBasedRecommender()
    
    # Train recommender
    print("Fitting recommender model...")
    recommender.fit(processed_data)
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(recommender, processed_data)
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return recommender, processed_data

def generate_recommendations(recommender, data, book_title=None, isbn=None, n=5):
    """
    Generate book recommendations
    
    Args:
        recommender: Trained recommender model
        data: DataFrame with book data
        book_title: Title of the book to base recommendations on
        isbn: ISBN of the book to base recommendations on
        n: Number of recommendations to return
    """
    if book_title:
        print(f"\nGenerating recommendations for book: '{book_title}'")
        recommendations = recommender.recommend(book_title, n=n)
    elif isbn:
        print(f"\nGenerating recommendations for ISBN: {isbn}")
        recommendations = recommender.get_recommendations_by_isbn(isbn, n=n)
    else:
        print("Error: Either book_title or isbn must be provided")
        return
    
    print("\nTop Recommendations:")
    for i, (_, book) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {book['title']} (Similarity: {book['similarity_score']:.4f})")
        print(f"   Author: {book.get('author', 'Unknown')}")
        print(f"   Type: {book.get('type', 'Unknown')}")
        print(f"   ISBN: {book.get('ISBN', 'Unknown')}")
        print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Book Recommendation Engine')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data collection parser
    collect_parser = subparsers.add_parser('collect', help='Collect data from Trove API')
    collect_parser.add_argument('--api-key', required=True, help='Trove API key')
    collect_parser.add_argument('--input', required=True, help='Input CSV file with ISBNs')
    collect_parser.add_argument('--output', required=True, help='Output CSV file for augmented data')
    
    # Data processing parser
    process_parser = subparsers.add_parser('process', help='Process data for recommendation engine')
    process_parser.add_argument('--input', required=True, help='Input CSV file with augmented data')
    process_parser.add_argument('--output', required=True, help='Output CSV file for processed data')
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train recommendation engine')
    train_parser.add_argument('--input', required=True, help='Input CSV file with processed data')
    
    # Recommendation parser
    recommend_parser = subparsers.add_parser('recommend', help='Generate book recommendations')
    recommend_parser.add_argument('--data', required=True, help='CSV file with processed data')
    recommend_parser.add_argument('--title', help='Book title to base recommendations on')
    recommend_parser.add_argument('--isbn', help='ISBN to base recommendations on')
    recommend_parser.add_argument('--n', type=int, default=5, help='Number of recommendations to return')
    
    # Pipeline parser
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the entire pipeline')
    pipeline_parser.add_argument('--api-key', required=True, help='Trove API key')
    pipeline_parser.add_argument('--input', required=True, help='Input CSV file with ISBNs')
    pipeline_parser.add_argument('--title', help='Book title to base recommendations on')
    pipeline_parser.add_argument('--isbn', help='ISBN to base recommendations on')
    pipeline_parser.add_argument('--n', type=int, default=5, help='Number of recommendations to return')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    if args.command == 'collect':
        collect_data(args.api_key, args.input, args.output)
    
    elif args.command == 'process':
        process_data(args.input, args.output)
    
    elif args.command == 'train':
        train_recommender(args.input)
    
    elif args.command == 'recommend':
        processed_data = load_data(args.data)
        recommender = ContentBasedRecommender()
        recommender.fit(processed_data)
        generate_recommendations(recommender, processed_data, args.title, args.isbn, args.n)
    
    elif args.command == 'pipeline':
        # Run the entire pipeline
        augmented_file = data_dir / 'augmented_data.csv'
        processed_file = data_dir / 'processed_data.csv'
        
        # Collect data
        collect_data(args.api_key, args.input, augmented_file)
        
        # Process data
        process_data(augmented_file, processed_file)
        
        # Train recommender
        recommender, data = train_recommender(processed_file)
        
        # Generate recommendations
        if args.title or args.isbn:
            generate_recommendations(recommender, data, args.title, args.isbn, args.n)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds") 