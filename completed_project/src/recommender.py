"""
Recommender Module - Implements content-based filtering for book recommendations
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Union, Any, Tuple
import os
import pickle
import joblib

class ContentBasedRecommender:
    """
    Content-based book recommendation system using TF-IDF and cosine similarity
    """
    
    def __init__(self, vectorizer=None):
        """
        Initialize the recommender.
        Insight: Zipf's Law - most words in a corpus are either super common (useless) or super rare (noise). The sweet spot is the middle frequencies.
        
        Args:
            vectorizer: Optional custom vectorizer, defaults to TfidfVectorizer
        """
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(
            stop_words='english',   # Theory: Remove noise words
            max_features=5000,     # Theory: Reduced from 15000 to focus on more relevant terms
            max_df=0.7,            # Theory: More restrictive - ignore terms in >70% of docs (was 0.85)
            min_df=1,              # Theory: Keep more rare terms (was 2)
            ngram_range=(1,2)      # Theory: Include bigrams for better context
        )
        self.tfidf_matrix = None
        self.books_df = None
        self.indices = None
        
        # Get the script directory for model persistence
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.dirname(self.script_dir)
        self.models_dir = os.path.join(self.project_dir, 'trained_models')
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the recommender to the dataset
        
        Args:
            df: DataFrame with processed book data including 'final_corpus' column from engineered data
        """
        # Store the DataFrame for later use
        self.books_df = df.copy()
        
        # Ensure we have a title column with unique values
        if 'title' not in df.columns:
            if 'Title' in df.columns:
                self.books_df['title'] = self.books_df['Title']
            else:
                # Use ISBN as title if no title column exists
                self.books_df['title'] = self.books_df['ISBN'].astype(str)
        
        # Handle duplicate titles by appending ISBN
        duplicate_mask = self.books_df['title'].duplicated(keep=False)
        self.books_df.loc[duplicate_mask, 'title'] = (
            self.books_df.loc[duplicate_mask, 'title'] + ' (ISBN: ' + 
            self.books_df.loc[duplicate_mask, 'ISBN'].astype(str) + ')'
        )
        
        # Create mapping from book titles to indices
        self.indices = pd.Series(self.books_df.index, index=self.books_df['title']).drop_duplicates()
        
        # Use the engineered final_corpus if available, otherwise fall back to original corpus
        corpus_column = 'final_corpus' if 'final_corpus' in self.books_df.columns else 'corpus'
        
        if corpus_column not in self.books_df.columns:
            raise ValueError(f"Neither 'final_corpus' nor 'corpus' column found in the dataset. Available columns: {list(self.books_df.columns)}")
        
        print(f"Using '{corpus_column}' column for feature extraction...")
        
        # Create TF-IDF matrix using the engineered corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books_df[corpus_column].fillna(''))
        
        print(f"Fitted recommender with {self.tfidf_matrix.shape[0]} books and {self.tfidf_matrix.shape[1]} features")
        print(f"Corpus column used: {corpus_column}")
    
    def save_model(self, model_name: str = "content_based_recommender") -> str:
        """
        Save the trained model to disk
        
        Args:
            model_name: Name for the saved model
            
        Returns:
            Path to the saved model file
        """
        if self.tfidf_matrix is None or self.books_df is None:
            raise ValueError("No trained model to save. Call fit() first.")
        
        # Create model data dictionary
        model_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'books_df': self.books_df,
            'indices': self.indices
        }
        
        # Save model using joblib (better for large numpy arrays)
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model_data, model_path)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_name: str = "content_based_recommender") -> bool:
        """
        Load a trained model from disk
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            # Load model data
            model_data = joblib.load(model_path)
            
            # Restore model components
            self.vectorizer = model_data['vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.books_df = model_data['books_df']
            self.indices = model_data['indices']
            
            print(f"Model loaded successfully from: {model_path}")
            print(f"Loaded model with {self.tfidf_matrix.shape[0]} books and {self.tfidf_matrix.shape[1]} features")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _get_similarity_scores(self, idx: int) -> np.ndarray:
        """
        Calculate similarity scores for a given book index
        
        Args:
            idx: Index of the book in the DataFrame
            
        Returns:
            Array of similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit() or load_model() first.")
            
        # Make sure idx is within bounds
        if idx < 0 or idx >= self.tfidf_matrix.shape[0]:
            raise IndexError(f"Index {idx} is out of bounds for matrix with shape {self.tfidf_matrix.shape}")
            
        # Get the similarity scores for the book at index idx
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx].reshape(1, -1), 
            self.tfidf_matrix
        ).flatten()
        
        return similarity_scores
    
    def _find_closest_titles(self, query: str, threshold: float = 0.3) -> List[str]:
        """
        Find titles that are closest to the query string
        
        Args:
            query: Search query
            threshold: Minimum similarity threshold
            
        Returns:
            List of closest title matches
        """
        if self.books_df is None:
            return []
        
        # Convert query to lowercase set of characters for fuzzy matching
        query_chars = set(query.lower())
        
        # Calculate character overlap ratio for each title
        similarities = []
        for title in self.books_df['title']:
            if pd.isna(title):
                similarities.append(0)
                continue
            
            title_chars = set(title.lower())
            
            # Skip if either set is empty
            if len(query_chars) == 0 or len(title_chars) == 0:
                similarities.append(0)
                continue
            
            # Calculate Jaccard similarity coefficient
            overlap = len(query_chars.intersection(title_chars))
            union = len(query_chars.union(title_chars))
            similarity = overlap / union if union > 0 else 0
            
            similarities.append(similarity)
        
        # Convert to numpy array for efficient operations
        similarities = np.array(similarities)
        
        # Get indices of titles above threshold, sorted by similarity
        above_threshold = similarities >= threshold
        if above_threshold.sum() > 0:
            indices = np.argsort(similarities)[::-1]
            above_indices = indices[similarities[indices] >= threshold]
            closest_titles = self.books_df.iloc[above_indices]['title'].tolist()
            return closest_titles[:5]  # Return top 5 closest matches
        
        return []
    
    def recommend(self, book_title: str, n: int = 5, min_similarity: float = 0.1) -> pd.DataFrame:
        """
        Generate book recommendations based on similarity to the given book
        
        Args:
            book_title: Title of the book to base recommendations on
            n: Number of recommendations to return
            min_similarity: Minimum similarity threshold for recommendations
            
        Returns:
            DataFrame with recommended books
        """
        if self.tfidf_matrix is None or self.books_df is None or self.indices is None:
            raise ValueError("Recommender has not been fitted yet. Call fit() or load_model() first.")
        
        # Check if the book title exists in our dataset
        if book_title not in self.indices:
            closest_titles = self._find_closest_titles(book_title)
            if not closest_titles:
                # If no close match, use a random book
                print(f"Book title '{book_title}' not found. Using a random book instead.")
                book_title = self.books_df['title'].sample(1).iloc[0]
            else:
                book_title = closest_titles[0]
                print(f"Using closest match: '{book_title}'")
        
        # Get the index of the book
        idx = self.indices[book_title]
        
        # Get similarity scores
        similarity_scores = self._get_similarity_scores(idx)
        
        # Filter by minimum similarity
        valid_mask = similarity_scores > min_similarity
        if valid_mask.sum() <= 1:  # Only query book
            return pd.DataFrame()  # No good recommendations
        
        # Sort and get top n+1 (including the book itself)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        top_indices = sorted_indices[1:n+1]  # Exclude the book itself
        
        # Create recommendations DataFrame
        recommendations = self.books_df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarity_scores[top_indices]
        
        return recommendations
    
    def get_recommendations_by_isbn(self, isbn: str, n: int = 5) -> pd.DataFrame:
        """
        Generate recommendations based on a book's ISBN
        
        Args:
            isbn: ISBN of the book to base recommendations on
            n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books
        """
        if self.books_df is None:
            raise ValueError("Recommender has not been fitted yet. Call fit() or load_model() first.")
        
        # Find the book with the given ISBN
        book = self.books_df[self.books_df['ISBN'] == isbn]
        
        if book.empty:
            raise ValueError(f"Book with ISBN '{isbn}' not found in the dataset.")
        
        # Get the title of the book
        book_title = book['title'].iloc[0]
        
        # Get recommendations based on the title
        return self.recommend(book_title, n)
    
    def get_recommendations_for_subject(self, subject: str, year_level: int, n: int = 5) -> pd.DataFrame:
        """
        Generate recommendations for a specific subject and year level
        
        Args:
            subject: Subject area (e.g., "ENGLISH", "MATHEMATICS")
            year_level: Year level (e.g., 1, 2, 3)
            n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books
        """
        if self.books_df is None:
            raise ValueError("Recommender has not been fitted yet. Call fit() or load_model() first.")
            
        # Filter books by subject and year level if those columns exist
        filtered_df = self.books_df
        
        if 'Subject' in self.books_df.columns:
            filtered_df = filtered_df[filtered_df['Subject'] == subject]
            
        if 'Year' in self.books_df.columns:
            filtered_df = filtered_df[filtered_df['Year'] == year_level]
            
        if filtered_df.empty:
            print(f"No books found for {subject} at Year {year_level}")
            return pd.DataFrame()
            
        # Return top n books
        return filtered_df.head(n) 