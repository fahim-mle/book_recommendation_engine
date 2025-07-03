"""
Data Processing Module - Handles text preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Optional, Union, Dict, List, Any

class TextProcessor:
    """
    Handles text preprocessing and feature engineering for book recommendation
    """
    
    def __init__(self):
        """Initialize the text processor"""
        pass
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data and create features for recommendation engine
        
        Args:
            df: DataFrame with book metadata
            
        Returns:
            DataFrame with processed features
        """
        df_features = df.copy()
        
        # Check if we have augmented data with Trove API fields
        has_augmented_data = 'Title' in df.columns
        
        # Extract publication year from 'Year' or 'issued'
        if 'issued' in df.columns:
            df_features['start_year'] = df_features['issued'].apply(self._extract_year)
        elif 'Year' in df.columns:
            df_features['start_year'] = df_features['Year']
        else:
            df_features['start_year'] = None
        
        # Process author names
        if 'author' in df.columns:
            df_features['author_processed'] = df_features['author'].apply(self._process_author)
        elif 'Contributors' in df.columns:
            df_features['author_processed'] = df_features['Contributors'].apply(self._process_author)
        else:
            df_features['author_processed'] = 'Unknown'
        
        # Process book type
        if 'type' in df.columns:
            df_features['type_processed'] = df_features['type'].apply(self._process_type)
        elif 'Type' in df.columns:
            df_features['type_processed'] = df_features['Type'].apply(self._process_type)
        else:
            df_features['type_processed'] = 'Unknown'
        
        # Create decade feature
        df_features['decade'] = df_features['start_year'].apply(self._get_decade)
        
        # Create title field if it doesn't exist
        if 'title' not in df_features.columns and 'Title' in df_features.columns:
            df_features['title'] = df_features['Title']
        elif 'title' not in df_features.columns and 'Title' not in df_features.columns:
            df_features['title'] = 'Unknown'
        
        # Create corpus feature (for text analysis later)
        df_features['corpus'] = df_features.apply(self._create_corpus, axis=1)
        
        # Calculate recency score (newer books get higher scores)
        df_features['recency_score'] = self._calculate_recency_scores(df_features)
        
        # Calculate popularity score based on relevance
        df_features['popularity_score'] = self._calculate_popularity_scores(df_features)
        
        return df_features
    
    def _extract_year(self, issued_str: Optional[str]) -> Optional[int]:
        """
        Extract publication year from 'issued' string
        
        Args:
            issued_str: String containing publication year information
            
        Returns:
            Extracted year as integer or None if not found
        """
        if pd.isna(issued_str):
            return None
        
        # Try to find a 4-digit year pattern
        year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(issued_str))
        if year_match:
            return int(year_match.group(1))
        
        return None
    
    def _process_author(self, author_str: Optional[str]) -> str:
        """
        Process author names to standardized format
        
        Args:
            author_str: Author name string
            
        Returns:
            Standardized author name
        """
        if pd.isna(author_str):
            return "Unknown"
        
        # Remove extra spaces and standardize
        return re.sub(r'\s+', ' ', str(author_str)).strip()
    
    def _process_type(self, type_str: Optional[str]) -> str:
        """
        Process book type to standardized format
        
        Args:
            type_str: Book type string
            
        Returns:
            Standardized book type
        """
        if pd.isna(type_str):
            return "Unknown"
        
        # Standardize book types
        type_lower = str(type_str).lower()
        if 'book' in type_lower:
            if 'illustrated' in type_lower:
                return "Book/Illustrated"
            elif 'audio' in type_lower:
                return "Audio Book"
            else:
                return "Book"
        elif 'article' in type_lower:
            return "Article"
        else:
            return type_str
    
    def _get_decade(self, year: Optional[int]) -> Optional[int]:
        """
        Get decade from year
        
        Args:
            year: Year as integer
            
        Returns:
            Decade as integer or None
        """
        if pd.isna(year):
            return None
        
        try:
            return int(year) // 10 * 10
        except (ValueError, TypeError):
            return None
    
    def _create_corpus(self, row: pd.Series) -> str:
        """
        Create corpus text from row data for text analysis
        
        Args:
            row: DataFrame row
            
        Returns:
            Combined text corpus
        """
        elements = []
        
        # Add ISBN as it's the most reliable identifier
        if not pd.isna(row.get('ISBN')):
            elements.append(str(row['ISBN']))
        
        # Add title if available
        if not pd.isna(row.get('title')):
            elements.append(str(row['title']))
        elif not pd.isna(row.get('Title')):
            elements.append(str(row['Title']))
            
        # Add author if available and not unknown
        if not pd.isna(row.get('author_processed')) and row.get('author_processed') != "Unknown":
            elements.append(str(row['author_processed']))
            
        # Add type if available and not unknown
        if not pd.isna(row.get('type_processed')) and row.get('type_processed') != "Unknown":
            elements.append(str(row['type_processed']))
        
        # Add subject if available
        if not pd.isna(row.get('Subject')):
            elements.append(str(row['Subject']))
            
        # Add subjects from Trove if available
        if not pd.isna(row.get('Subjects')):
            elements.append(str(row['Subjects']))
            
        # Add abstract if available
        if not pd.isna(row.get('Abstract')):
            elements.append(str(row['Abstract']))
        
        # Join all elements
        return ' '.join(elements)
    
    def _calculate_recency_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate recency scores (newer books get higher scores)
        
        Args:
            df: DataFrame with start_year column
            
        Returns:
            Series with recency scores
        """
        current_year = datetime.now().year
        
        # Safely calculate max_age
        if len(df) == 0 or df['start_year'].count() == 0:
            max_age = 100  # Default if no valid years
        else:
            valid_years = df['start_year'].dropna()
            if len(valid_years) == 0:
                max_age = 100  # Default if no valid years
            else:
                try:
                    min_year = valid_years.min()
                    if pd.isna(min_year):
                        max_age = 100
                    else:
                        max_age = current_year - min_year
                except:
                    max_age = 100
        
        def calculate_recency(year):
            if pd.isna(year):
                return 0.5  # Default value for unknown years
            try:
                age = current_year - int(year)
                # Normalize to 0-1 range, with newer books closer to 1
                return 1 - (age / max_age) if max_age > 0 else 0.5
            except (ValueError, TypeError):
                return 0.5
        
        return df['start_year'].apply(calculate_recency)
    
    def _calculate_popularity_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate popularity scores based on relevance
        
        Args:
            df: DataFrame with relevance_score column
            
        Returns:
            Series with popularity scores
        """
        if 'relevance_score' in df.columns:
            # Normalize relevance score to 0-1 range
            max_relevance = df['relevance_score'].max()
            min_relevance = df['relevance_score'].min()
            
            def normalize_relevance(score):
                if pd.isna(score) or max_relevance == min_relevance:
                    return 0.5  # Default for unknown or uniform relevance
                return (score - min_relevance) / (max_relevance - min_relevance)
            
            return df['relevance_score'].apply(normalize_relevance)
        else:
            return pd.Series([0.5] * len(df))  # Default if relevance_score not available 