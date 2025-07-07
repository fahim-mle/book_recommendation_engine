"""
Usage script for the trained content-based book recommender
"""
import pandas as pd
import os
import sys
import numpy as np

# Add the src directory to the path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from recommender import ContentBasedRecommender

def load_and_use_recommender():
    """
    Load the trained model and demonstrate its usage
    """
    print("📚 BOOK RECOMMENDER - USAGE DEMO")
    print("=" * 50)
    
    # Initialize recommender
    recommender = ContentBasedRecommender()
    
    # Load the trained model - try improved model first, then fall back
    print("🔄 Loading trained model...")
    if os.path.exists(os.path.join(recommender.models_dir, "content_based_recommender_improved.joblib")):
        model_name = "content_based_recommender_improved"
        if not recommender.load_model(model_name):
            print("❌ Failed to load improved model. Trying engineered model...")
            model_name = "content_based_recommender_engineered"
            if not recommender.load_model(model_name):
                print("❌ Failed to load model. Please run train_recommender.py first.")
                return
    else:
        model_name = "content_based_recommender_engineered"
        if not recommender.load_model(model_name):
            print("❌ Failed to load model. Please run train_recommender.py first.")
            return
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model contains {0 if recommender.books_df is None else len(recommender.books_df)} books")
    print(f"📊 Model type: {model_name}")
    
    # Show some sample books
    print("\n📖 Sample books in the dataset:")
    try:
        sample_books = recommender.books_df[['title', 'Subject', 'Year']].sample(5)
        for i, (_, book) in enumerate(sample_books.iterrows(), 1):
            # Handle NaN values for title, Subject and Year
            title = "Unknown title"
            if pd.notnull(book.get('title')):
                title = str(book.get('title', ''))
            title_display = title[:60] + "..." if len(title) > 60 else title
            
            subject = "Unknown"
            if pd.notnull(book.get('Subject')):
                subject = str(book.get('Subject', ''))
                
            year = "Unknown"
            if pd.notnull(book.get('Year')):
                year = str(book.get('Year', ''))
                
            print(f"  {i}. {title_display} (Subject: {subject}, Year: {year})")
    except Exception as e:
        print(f"⚠️ Error displaying sample books: {e}")
    
    # Interactive recommendation demo
    print("\n🎯 RECOMMENDATION DEMO")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations by book title")
        print("2. Get recommendations by ISBN")
        print("3. Get recommendations for subject and year")
        print("4. Show random book recommendations")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            book_title = input("Enter a book title (or part of it): ").strip()
            if book_title:
                try:
                    recommendations = recommender.recommend(book_title, n=5, min_similarity=0.1)
                    if recommendations.empty:
                        print(f"\n❌ No good recommendations found for '{book_title}' (similarity too low)")
                    else:
                        print(f"\n🎯 Top {len(recommendations)} recommendations for '{book_title}':")
                        for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                            # Handle NaN values
                            title = "Unknown title"
                            if pd.notnull(book.get('title')):
                                title = str(book.get('title', ''))
                                
                            subject = "Unknown"
                            if pd.notnull(book.get('Subject')):
                                subject = str(book.get('Subject', ''))
                                
                            year = "Unknown"
                            if pd.notnull(book.get('Year')):
                                year = str(book.get('Year', ''))
                                
                            print(f"  {i}. {title}")
                            print(f"     Subject: {subject}, Year: {year}")
                            print(f"     Similarity: {book['similarity_score']:.3f}")
                            print()
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == '2':
            isbn = input("Enter an ISBN: ").strip()
            if isbn:
                try:
                    recommendations = recommender.get_recommendations_by_isbn(isbn, n=5)
                    if recommendations.empty:
                        print(f"\n❌ No good recommendations found for ISBN {isbn}")
                    else:
                        print(f"\n🎯 Top {len(recommendations)} recommendations for ISBN {isbn}:")
                        for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                            # Handle NaN values
                            title = "Unknown title"
                            if pd.notnull(book.get('title')):
                                title = str(book.get('title', ''))
                                
                            subject = "Unknown"
                            if pd.notnull(book.get('Subject')):
                                subject = str(book.get('Subject', ''))
                                
                            year = "Unknown"
                            if pd.notnull(book.get('Year')):
                                year = str(book.get('Year', ''))
                                
                            print(f"  {i}. {title}")
                            print(f"     Subject: {subject}, Year: {year}")
                            print(f"     Similarity: {book['similarity_score']:.3f}")
                            print()
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == '3':
            subject = input("Enter subject (e.g., ENGLISH, MATHEMATICS): ").strip().upper()
            try:
                year = int(input("Enter year level (0-12): ").strip())
                recommendations = recommender.get_recommendations_for_subject(subject, year, n=5)
                if recommendations.empty:
                    print(f"❌ No books found for {subject} at Year {year}")
                else:
                    print(f"\n📚 Top {len(recommendations)} {subject} books for Year {year}:")
                    for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                        # Handle NaN values
                        title = "Unknown title"
                        if pd.notnull(book.get('title')):
                            title = str(book.get('title', ''))
                            
                        print(f"  {i}. {title}")
                        if 'quality_score' in book and pd.notnull(book.get('quality_score')):
                            print(f"     Quality Score: {book.get('quality_score', 0):.3f}")
                        print()
            except ValueError:
                print("❌ Please enter a valid year number")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '4':
            try:
                # Find a non-NaN title for the random book
                if recommender.books_df is None:
                    print("❌ No data available")
                    continue
                    
                valid_titles = recommender.books_df.dropna(subset=['title'])
                if len(valid_titles) == 0:
                    print("❌ No books with valid titles found in the dataset")
                    continue
                    
                random_book = valid_titles['title'].sample(1).iloc[0]
                recommendations = recommender.recommend(random_book, n=5, min_similarity=0.1)
                if recommendations.empty:
                    print(f"\n❌ No good recommendations found for random book (similarity too low)")
                    print(f"🎲 Random book: {random_book}")
                else:
                    print(f"\n🎲 Random book: {random_book}")
                    print(f"🎯 Top {len(recommendations)} similar books:")
                    for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                        # Handle NaN values
                        title = "Unknown title"
                        if pd.notnull(book.get('title')):
                            title = str(book.get('title', ''))
                            
                        subject = "Unknown"
                        if pd.notnull(book.get('Subject')):
                            subject = str(book.get('Subject', ''))
                            
                        year = "Unknown"
                        if pd.notnull(book.get('Year')):
                            year = str(book.get('Year', ''))
                            
                        print(f"  {i}. {title}")
                        print(f"     Subject: {subject}, Year: {year}")
                        print(f"     Similarity: {book['similarity_score']:.3f}")
                        print()
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    load_and_use_recommender() 