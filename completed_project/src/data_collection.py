"""
Data Collection Module - Trove API v3 integration (following MA5851_Assessment1.ipynb reference)
"""
import requests
import pandas as pd
import time
import logging
from typing import List, Dict, Optional

# --- Setup Logging ---
logging.basicConfig(filename='data/failed_isbns.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class APIClient:
    """
    Client for interacting with the Trove API v3 (with retries, error handling, and logging)
    """
    def __init__(self, api_key: str = "doqGm0j0QXuHYDcZHV79KuJaDV8aeC1Y"):
        self.api_key = api_key
        self.base_url = "https://api.trove.nla.gov.au/v3/result"
        self.headers = {"X-API-KEY": self.api_key}

    def get_book_details(self, isbn: str) -> Optional[List[Dict]]:
        """
        Fetches book details from the Trove API v3 with error handling,
        retries, and exponential backoff.
        """
        params = {
            'q': f'isbn:{isbn}',
            'category': 'book',
            'encoding': 'json'
        }
        retries = 3
        backoff_factor = 2
        for attempt in range(retries):
            try:
                print(f"Fetching details for ISBN: {isbn} (Attempt {attempt + 1})")
                response = requests.get(self.base_url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                if data.get('category', [{}])[0].get('records', {}).get('work'):
                    print(f"--> SUCCESS: Found details for ISBN: {isbn}")
                    return data['category'][0]['records']['work']
                else:
                    print(f"--> INFO: No work found for ISBN: {isbn}")
                    return None
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = backoff_factor ** attempt
                    print(f"--> WARNING: Rate limit exceeded for ISBN {isbn}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"--> ERROR: HTTP error for ISBN {isbn}: {e}")
                    logging.error(f"Failed to fetch ISBN {isbn} due to HTTP error: {e.response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"--> ERROR: Request failed for ISBN {isbn}: {e}")
                logging.error(f"Failed to fetch ISBN {isbn} due to RequestException: {e}")
                return None
            except (KeyError, IndexError) as e:
                print(f"--> ERROR: Unexpected API response for ISBN {isbn}: {e}")
                logging.error(f"Failed to fetch ISBN {isbn} due to unexpected API response: {e}")
                return None
        print(f"--> FAILURE: All retries failed for ISBN {isbn}.")
        logging.error(f"Failed to fetch ISBN {isbn} after {retries} retries.")
        return None

    def augment_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment book dataframe with additional data from Trove API v3
        Args:
            df: DataFrame with book data (must contain 'ISBN' column)
        Returns:
            Augmented DataFrame
        """
        if 'ISBN' not in df.columns:
            raise ValueError("DataFrame must contain 'ISBN' column")
        unique_isbns = df['ISBN'].unique()
        all_book_rows = []
        for isbn in unique_isbns:
            works = self.get_book_details(isbn)
            if works:
                for work in works:
                    trove_id = work.get('id')
                    title = work.get('title')
                    contributor_list = work.get('contributor')
                    contributor = contributor_list[0] if contributor_list else ''
                    issued = work.get('issued')
                    work_type = ', '.join(work.get('type', []))
                    relevance_score = work.get('relevance', {}).get('score')
                    relevance_value = work.get('relevance', {}).get('value')
                    identifiers = work.get('identifier', [])
                    if identifiers:
                        for identifier in identifiers:
                            row = {
                                'ISBN': isbn,
                                'trove_id': trove_id,
                                'title': title,
                                'author': contributor,
                                'issued': issued,
                                'type': work_type,
                                'relevance_score': relevance_score,
                                'relevance_value': relevance_value,
                                'identifier_type': identifier.get('type'),
                                'identifier_linktype': identifier.get('linktype'),
                                'identifier_value': identifier.get('value')
                            }
                            all_book_rows.append(row)
                    else:
                        row = {
                            'ISBN': isbn,
                            'trove_id': trove_id,
                            'title': title,
                            'author': contributor,
                            'issued': issued,
                            'type': work_type,
                            'relevance_score': relevance_score,
                            'relevance_value': relevance_value,
                            'identifier_type': None,
                            'identifier_linktype': None,
                            'identifier_value': None
                        }
                        all_book_rows.append(row)
            else:
                # Output a row with only ISBN and empty columns if no data found
                row = {
                    'ISBN': isbn,
                    'trove_id': None,
                    'title': None,
                    'author': None,
                    'issued': None,
                    'type': None,
                    'relevance_score': None,
                    'relevance_value': None,
                    'identifier_type': None,
                    'identifier_linktype': None,
                    'identifier_value': None
                }
                all_book_rows.append(row)
            time.sleep(0.01)  # Throttle API calls
        columns_list = [
            'ISBN', 'trove_id', 'title', 'author', 'issued', 'type',
            'relevance_score', 'relevance_value', 'identifier_type',
            'identifier_linktype', 'identifier_value'
        ]
        detailed_df = pd.DataFrame(all_book_rows, columns=columns_list)
        return detailed_df

if __name__ == "__main__":
    import sys
    import os

    # Example usage: python src/data_collection.py input.csv output.csv
    if len(sys.argv) != 3:
        print("Usage: python src/data_collection.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    if not os.path.exists(input_csv):
        print(f"Input file {input_csv} does not exist.")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    client = APIClient()
    augmented = client.augment_dataset(df)
    augmented.to_csv(output_csv, index=False)
    print(f"Augmented data saved to {output_csv}") 