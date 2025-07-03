"""
Google Books Data Collection Module - Simple ISBN-based book data collection
"""
import requests
import pandas as pd
import time
import logging
from typing import Dict, Optional

class GoogleBooksAPIClient:
    """
    Simple client for Google Books API using ISBN lookups
    """
    
    def __init__(self):
        self.base_url = "https://www.googleapis.com/books/v1/volumes"
        self.session = requests.Session()
        
    def get_book_details(self, isbn: str) -> Optional[Dict]:
        """
        Get book details from Google Books API using ISBN
        
        Args:
            isbn: ISBN of the book
            
        Returns:
            Book metadata dictionary or None if not found
        """
        params = {
            'q': f'isbn:{isbn}',
            'maxResults': 1
        }
        
        try:
            print(f"Fetching: {isbn}")
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('totalItems', 0) > 0:
                volume_info = data['items'][0]['volumeInfo']
                return self._extract_metadata(isbn, volume_info)
            else:
                print(f"Not found: {isbn}")
                return None
                
        except Exception as e:
            print(f"Error for {isbn}: {e}")
            return None
        finally:
            time.sleep(0.01)  # Rate limiting
    
    def _extract_metadata(self, isbn: str, volume_info: Dict) -> Dict:
        """Extract the good stuff from API response"""
        authors = volume_info.get('authors', [])
        categories = volume_info.get('categories', [])
        
        return {
            'ISBN': isbn,
            'title': volume_info.get('title'),
            'subtitle': volume_info.get('subtitle'),
            'authors': ', '.join(authors) if authors else None,
            'publisher': volume_info.get('publisher'),
            'published_date': volume_info.get('publishedDate'),
            'description': volume_info.get('description'),
            'page_count': volume_info.get('pageCount'),
            'categories': ', '.join(categories) if categories else None,
            'average_rating': volume_info.get('averageRating'),
            'ratings_count': volume_info.get('ratingsCount'),
            'language': volume_info.get('language')
        }
    
    def augment_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Google Books data to your ISBN dataset
        
        Args:
            df: DataFrame with 'ISBN' column
            
        Returns:
            DataFrame with Google Books metadata
        """
        if 'ISBN' not in df.columns:
            raise ValueError("Need 'ISBN' column in DataFrame")
            
        unique_isbns = df['ISBN'].unique()
        all_books = []
        
        for i, isbn in enumerate(unique_isbns):
            print(f"{i+1}/{len(unique_isbns)}")
            
            book_data = self.get_book_details(isbn)
            
            if book_data:
                all_books.append(book_data)
            else:
                # Empty row for failed lookups
                all_books.append({
                    'ISBN': isbn,
                    'title': None,
                    'subtitle': None,
                    'authors': None,
                    'publisher': None,
                    'published_date': None,
                    'description': None,
                    'page_count': None,
                    'categories': None,
                    'average_rating': None,
                    'ratings_count': None,
                    'language': None
                })
        
        result_df = pd.DataFrame(all_books)
        
        print(f"\nDone! Found data for {result_df['title'].notna().sum()}/{len(unique_isbns)} books")
        print(f"Books with descriptions: {result_df['description'].notna().sum()}")
        
        return result_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python google_books_client.py input.csv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    df = pd.read_csv(input_file)
    client = GoogleBooksAPIClient()
    result = client.augment_dataset(df)
    result.to_csv(output_file, index=False)
    
    print(f"Saved to {output_file}")