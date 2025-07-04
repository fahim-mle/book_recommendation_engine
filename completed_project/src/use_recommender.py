"""
Usage script for the trained content-based book recommender
"""
import pandas as pd
import os
import sys

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
    
    # Load the trained model
    print("🔄 Loading trained model...")
    if not recommender.load_model("content_based_recommender_engineered"):
        print("❌ Failed to load model. Please run train_recommender.py first.")
        return
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model contains {len(recommender.books_df)} books")
    
    # Show some sample books
    print("\n📖 Sample books in the dataset:")
    sample_books = recommender.books_df[['title', 'Subject', 'Year']].sample(5)
    for i, (_, book) in enumerate(sample_books.iterrows(), 1):
        print(f"  {i}. {book['title'][:60]}... (Subject: {book['Subject']}, Year: {book['Year']})")
    
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
                    recommendations = recommender.recommend(book_title, n=5)
                    print(f"\n🎯 Top 5 recommendations for '{book_title}':")
                    for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                        print(f"  {i}. {book['title']}")
                        print(f"     Subject: {book['Subject']}, Year: {book['Year']}")
                        print(f"     Similarity: {book['similarity_score']:.3f}")
                        print()
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == '2':
            isbn = input("Enter an ISBN: ").strip()
            if isbn:
                try:
                    recommendations = recommender.get_recommendations_by_isbn(isbn, n=5)
                    print(f"\n🎯 Top 5 recommendations for ISBN {isbn}:")
                    for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                        print(f"  {i}. {book['title']}")
                        print(f"     Subject: {book['Subject']}, Year: {book['Year']}")
                        print(f"     Similarity: {book['similarity_score']:.3f}")
                        print()
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == '3':
            subject = input("Enter subject (e.g., ENGLISH, MATHEMATICS): ").strip().upper()
            try:
                year = int(input("Enter year level (0-12): ").strip())
                recommendations = recommender.get_recommendations_for_subject(subject, year, n=5)
                if not recommendations.empty:
                    print(f"\n📚 Top 5 {subject} books for Year {year}:")
                    for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                        print(f"  {i}. {book['title']}")
                        if 'quality_score' in book:
                            print(f"     Quality Score: {book['quality_score']:.3f}")
                        print()
                else:
                    print(f"❌ No books found for {subject} at Year {year}")
            except ValueError:
                print("❌ Please enter a valid year number")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '4':
            try:
                random_book = recommender.books_df['title'].sample(1).iloc[0]
                recommendations = recommender.recommend(random_book, n=5)
                print(f"\n🎲 Random book: {random_book}")
                print("🎯 Top 5 similar books:")
                for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                    print(f"  {i}. {book['title']}")
                    print(f"     Subject: {book['Subject']}, Year: {book['Year']}")
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