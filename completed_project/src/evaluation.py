"""
Evaluation Module - Evaluates the performance of the recommendation system
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Any, Optional, Union

class ModelEvaluator:
    """
    Evaluates the performance of book recommendation models
    """
    
    def __init__(self):
        """Initialize the model evaluator"""
        pass
    
    def evaluate(self, recommender, data: pd.DataFrame, test_size: float = 0.2, 
                 random_state: int = 42) -> Dict[str, float]:
        """
        Evaluate the recommender model using train-test split
        
        Args:
            recommender: Trained recommender model
            data: DataFrame with book data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Split data into train and test sets
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        
        # Reset index to avoid index mismatch errors
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        
        # Fit the recommender on the training data
        recommender.fit(train_data)
        
        # Evaluate on the test data
        metrics = self._calculate_metrics(recommender, test_data)
        
        return metrics
    
    def _calculate_metrics(self, recommender, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate evaluation metrics for the recommender
        
        Args:
            recommender: Trained recommender model
            test_data: Test data to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        # For each book in the test set, get recommendations
        # and compare with actual data to calculate metrics
        
        # In a real implementation, we would need ground truth data
        # about which books are actually relevant to each other.
        # Since we don't have that, we'll use proxy metrics:
        
        # 1. Sample a subset of books for evaluation
        sample_size = min(50, len(test_data))
        sample_books = test_data.sample(sample_size).reset_index(drop=True)
        
        # 2. Calculate metrics
        metrics = {
            'coverage': self._calculate_coverage(recommender, test_data),
            'diversity': self._calculate_diversity(recommender, sample_books),
            'novelty': self._calculate_novelty(recommender, sample_books, test_data),
            'avg_similarity_score': self._calculate_avg_similarity(recommender, sample_books)
        }
        
        return metrics
    
    def _calculate_coverage(self, recommender, test_data: pd.DataFrame, n: int = 5) -> float:
        """
        Calculate catalog coverage: percentage of books that get recommended
        
        Args:
            recommender: Trained recommender model
            test_data: Test data
            n: Number of recommendations per book
            
        Returns:
            Coverage metric (0-1)
        """
        # Get all unique book titles
        all_titles = set(test_data['title'].dropna())
        
        # Sample a subset of books to generate recommendations for
        sample_size = min(20, len(test_data))
        sample_books = test_data.sample(sample_size).reset_index(drop=True)
        
        # Track which books get recommended
        recommended_titles = set()
        
        # For each book, get recommendations and add to the set
        for _, book in sample_books.iterrows():
            if pd.isna(book['title']):
                continue
                
            try:
                recommendations = recommender.recommend(book['title'], n=n)
                recommended_titles.update(recommendations['title'].tolist())
            except (ValueError, KeyError):
                # Skip books that cause errors
                continue
        
        # Calculate coverage
        if not all_titles:
            return 0.0
            
        coverage = len(recommended_titles) / len(all_titles)
        return coverage
    
    def _calculate_diversity(self, recommender, sample_books: pd.DataFrame, n: int = 5) -> float:
        """
        Calculate diversity: average dissimilarity between recommendations
        
        Args:
            recommender: Trained recommender model
            sample_books: Sample of books to evaluate
            n: Number of recommendations per book
            
        Returns:
            Diversity metric (0-1)
        """
        diversity_scores = []
        
        # For each book, get recommendations and calculate diversity
        for _, book in sample_books.iterrows():
            if pd.isna(book['title']):
                continue
                
            try:
                recommendations = recommender.recommend(book['title'], n=n)
                
                # If we have similarity scores, calculate diversity as 1 - avg_similarity
                if 'similarity_score' in recommendations.columns:
                    avg_similarity = recommendations['similarity_score'].mean()
                    diversity = 1 - avg_similarity
                    diversity_scores.append(diversity)
            except (ValueError, KeyError):
                # Skip books that cause errors
                continue
        
        # Calculate average diversity
        if not diversity_scores:
            return 0.0
            
        avg_diversity = sum(diversity_scores) / len(diversity_scores)
        return avg_diversity
    
    def _calculate_novelty(self, recommender, sample_books: pd.DataFrame, 
                          all_data: pd.DataFrame, n: int = 5) -> float:
        """
        Calculate novelty: how surprising/unexpected the recommendations are
        
        Args:
            recommender: Trained recommender model
            sample_books: Sample of books to evaluate
            all_data: Complete dataset
            n: Number of recommendations per book
            
        Returns:
            Novelty metric (0-1)
        """
        # Calculate popularity of each book
        book_counts = all_data['title'].value_counts()
        total_books = len(all_data)
        
        # Convert to probability
        book_probs = book_counts / total_books
        
        novelty_scores = []
        
        # For each book, get recommendations and calculate novelty
        for _, book in sample_books.iterrows():
            if pd.isna(book['title']):
                continue
                
            try:
                recommendations = recommender.recommend(book['title'], n=n)
                
                # Calculate novelty as negative log probability (information content)
                rec_titles = recommendations['title'].tolist()
                rec_probs = [book_probs.get(title, 1/total_books) for title in rec_titles]
                
                # Higher is more novel (less popular)
                rec_novelties = [-np.log2(prob) for prob in rec_probs]
                
                if rec_novelties:
                    avg_novelty = sum(rec_novelties) / len(rec_novelties)
                    # Normalize to 0-1 range
                    normalized_novelty = min(1.0, avg_novelty / 10.0)
                    novelty_scores.append(normalized_novelty)
            except (ValueError, KeyError):
                # Skip books that cause errors
                continue
        
        # Calculate average novelty
        if not novelty_scores:
            return 0.0
            
        avg_novelty = sum(novelty_scores) / len(novelty_scores)
        return avg_novelty
    
    def _calculate_avg_similarity(self, recommender, sample_books: pd.DataFrame, n: int = 5) -> float:
        """
        Calculate average similarity score of recommendations
        
        Args:
            recommender: Trained recommender model
            sample_books: Sample of books to evaluate
            n: Number of recommendations per book
            
        Returns:
            Average similarity score (0-1)
        """
        similarity_scores = []
        
        # For each book, get recommendations and extract similarity scores
        for _, book in sample_books.iterrows():
            if pd.isna(book['title']):
                continue
                
            try:
                recommendations = recommender.recommend(book['title'], n=n)
                
                if 'similarity_score' in recommendations.columns:
                    scores = recommendations['similarity_score'].tolist()
                    similarity_scores.extend(scores)
            except (ValueError, KeyError):
                # Skip books that cause errors
                continue
        
        # Calculate average similarity
        if not similarity_scores:
            return 0.0
            
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        return avg_similarity 