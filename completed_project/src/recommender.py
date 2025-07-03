"""
Recommender Module - Implements content-based filtering for book recommendations
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Union, Any, Tuple

class ContentBasedRecommender:
    """
    Content-based book recommendation system using TF-IDF and cosine similarity
    """
    
    def __init__(self, vectorizer=None):
        """
        Initialize the recommender
        
        Args:
            vectorizer: Optional custom vectorizer, defaults to TfidfVectorizer
        """
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            max_df=0.85,
            min_df=2
        )
        self.tfidf_matrix = None
        self.books_df = None
        self.indices = None
        
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the recommender to the dataset
        
        Args:
            df: DataFrame with processed book data including 'corpus' column
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
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books_df['corpus'].fillna(''))
        
        print(f"Fitted recommender with {self.tfidf_matrix.shape[0]} books and {self.tfidf_matrix.shape[1]} features")
    
    def _get_similarity_scores(self, idx: int) -> np.ndarray:
        """
        Calculate similarity scores for a given book index
        
        Args:
            idx: Index of the book in the DataFrame
            
        Returns:
            Array of similarity scores
        """
        # Make sure idx is within bounds
        if idx < 0 or idx >= self.tfidf_matrix.shape[0]:
            raise IndexError(f"Index {idx} is out of bounds for matrix with shape {self.tfidf_matrix.shape}")
            
        # Get the similarity scores for the book at index idx
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx].reshape(1, -1), 
            self.tfidf_matrix
        ).flatten()
        
        return similarity_scores
    
    def recommend(self, book_title: str, n: int = 5) -> pd.DataFrame:
        """
        Generate book recommendations based on similarity to the given book
        
        Args:
            book_title: Title of the book to base recommendations on
            n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books
        """
        if self.tfidf_matrix is None or self.books_df is None:
            raise ValueError("Recommender has not been fitted yet. Call fit() first.")
        
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
        
        # Get the indices of the top n similar books (excluding the book itself)
        similar_indices = similarity_scores.argsort()[::-1][1:n+1]
        
        # Get the data for the recommended books
        recommendations = self.books_df.iloc[similar_indices].copy()
        
        # Add similarity score to the recommendations
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        
        # Sort by similarity score
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        return recommendations
    
    def _find_closest_titles(self, query: str, threshold: float = 0.3) -> List[str]:
        """
        Find the closest matching titles for a query
        
        Args:
            query: The search query
            threshold: Minimum similarity threshold
            
        Returns:
            List of closest matching titles
        """
        # Convert query to lowercase for case-insensitive matching
        query = query.lower()
        
        # Calculate similarity between query and all titles
        similarities = []
        for title in self.books_df['title'].dropna():
            # Simple string similarity using character overlap
            title_lower = title.lower()
            
            # Calculate Jaccard similarity between character sets
            query_chars = set(query)
            title_chars = set(title_lower)
            
            if not query_chars or not title_chars:
                similarities.append(0)
                continue
                
            intersection = len(query_chars.intersection(title_chars))
            union = len(query_chars.union(title_chars))
            
            jaccard = intersection / union if union > 0 else 0
            
            # Check if query is a substring of title
            contains_bonus = 0.3 if query in title_lower else 0
            
            # Final similarity score
            similarity = jaccard + contains_bonus
            similarities.append(similarity)
        
        # Get titles with similarity above threshold
        titles = self.books_df['title'].dropna().values
        close_matches = [(titles[i], similarities[i]) for i in range(len(titles)) if similarities[i] >= threshold]
        
        # Sort by similarity (descending)
        close_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the titles
        return [match[0] for match in close_matches]
    
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
            raise ValueError("Recommender has not been fitted yet. Call fit() first.")
        
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
            raise ValueError("Recommender has not been fitted yet. Call fit() first.")
            
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