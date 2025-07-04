import pandas as pd
import numpy as np
import os

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # Go up one level to completed_project
data_dir = os.path.join(project_dir, 'data')

class TextProcessor:
    """
    Text processor for the book recommendation engine
    """
    def __init__(self):
        """Initialize the text processor"""
        pass
        
    def preprocess(self, df):
        """
        Process and enhance text data for the recommendation engine
        
        Args:
            df: DataFrame with book data
            
        Returns:
            Enhanced DataFrame with additional text features
        """
        return engineer_educational_features(df)

def create_better_corpus(row):
    """
    Create an improved corpus with better feature weighting
    
    Args:
        row: DataFrame row
        
    Returns:
        Improved corpus string
    """
    parts = []
    
    # Subject: 3x weight (not 6x)
    if pd.notna(row['Subject']):
        parts.extend([str(row['Subject']).lower()] * 3)
    
    # Year: 2x weight 
    if pd.notna(row['Year']):
        parts.extend([f"year {row['Year']}"] * 2)
    
    # Title: 2x weight
    if pd.notna(row['title']):
        parts.extend([str(row['title']).lower()] * 2)
    
    # Description: 1x weight (full text)
    if pd.notna(row['description']):
        parts.append(str(row['description'])[:500].lower())
    
    return ' '.join(parts)

def engineer_educational_features(df):
    """
    Process and enhance text data for the recommendation engine
    
    Args:
        df: DataFrame with book data
        
    Returns:
        DataFrame with engineered features
    """
    print("🔧 Engineering educational features...")
    
    # Create corpus column if it doesn't exist
    if 'corpus' not in df.columns:
        # Create basic corpus from available fields
        print("📝 Creating text corpus...")
        df['corpus'] = df.apply(
            lambda row: ' '.join(filter(None, [
                str(row['title']) if pd.notna(row['title']) else None,
                str(row['author']) if pd.notna(row['author']) else None,
                str(row['Subject']) if pd.notna(row['Subject']) else None,
                str(row['description']) if pd.notna(row['description']) else None
            ])), axis=1
        )
    
    # Extract educational information
    print("📚 Extracting educational information...")
    
    # Create grade category (Primary/Secondary/Higher)
    df['grade_category'] = df['Year'].apply(grade_category)
    
    # Extract subject keywords
    df['subject_keywords'] = df['Subject'].apply(extract_subject_keywords)
    
    # Create enhanced corpus with weighted fields
    print("🔄 Creating enhanced corpus with educational context...")
    df['enhanced_corpus'] = df.apply(create_enhanced_corpus, axis=1)
    
    # Create better corpus with improved weighting
    print("⭐ Creating better corpus with improved weighting...")
    df['final_corpus'] = df.apply(create_better_corpus, axis=1)
    
    # Calculate book quality score
    print("⭐ Calculating book quality scores...")
    df['quality_score'] = calculate_quality_scores(df)
    
    print(f"✅ Educational feature engineering complete. Dataset shape: {df.shape}")
    
    return df

def grade_category(year):
    """
    Categorize grade level
    
    Args:
        year: Year level
        
    Returns:
        Grade category string
    """
    if pd.isna(year):
        return "unknown"
    
    year = int(year)
    if year <= 2:
        return "early_primary foundation prep"
    elif year <= 6:
        return "primary elementary"
    elif year <= 10:
        return "junior_secondary middle"
    else:
        return "senior_secondary vce_hsc"

def extract_subject_keywords(subject):
    """
    Extract keywords for a subject
    
    Args:
        subject: Subject name
        
    Returns:
        Subject keywords string
    """
    if pd.isna(subject):
        return ""
    
    subject = str(subject).upper()
    keywords = []
    
    if any(term in subject for term in ['MATH', 'ALGEBRA', 'CALCULUS', 'GEOMETRY']):
        keywords = ['mathematics', 'algebra', 'calculus', 'geometry', 'statistics', 'numerical']
    elif any(term in subject for term in ['SCIENCE', 'BIOLOGY', 'CHEMISTRY', 'PHYSICS']):
        keywords = ['science', 'biology', 'chemistry', 'physics', 'laboratory', 'scientific']
    elif 'ENGLISH' in subject:
        keywords = ['english', 'literature', 'writing', 'reading', 'language', 'literacy']
    elif 'HISTORY' in subject:
        keywords = ['history', 'historical', 'social', 'politics', 'civilization']
    elif any(term in subject for term in ['BUSINESS', 'ECONOMICS']):
        keywords = ['business', 'economics', 'commerce', 'management', 'finance']
    elif any(term in subject for term in ['ART', 'MUSIC', 'DRAMA']):
        keywords = ['arts', 'creative', 'artistic', 'performance', 'visual']
    
    return ' '.join(keywords)

def create_enhanced_corpus(row):
    """
    Create enhanced corpus with educational context
    
    Args:
        row: DataFrame row
        
    Returns:
        Enhanced corpus string
    """
    parts = []
    
    # Subject gets massive weight (most important)
    if pd.notna(row['Subject']):
        subject_clean = str(row['Subject']).lower().strip()
        parts.extend([subject_clean] * 6)
    
    # Year level critical for age-appropriate content
    if pd.notna(row['Year']):
        year_terms = f"year {row['Year']} grade {row['Year']} level {row['Year']}"
        parts.extend([year_terms] * 4)
    
    # Google Books categories
    if pd.notna(row['categories']):
        categories = str(row['categories']).lower()
        parts.extend([categories] * 3)
    
    # Title (prefer Google Books title, then Trove title, then original title)
    if pd.notna(row.get('title_google')) and str(row.get('title_google')) != 'nan':
        title = str(row.get('title_google')).lower()
    elif pd.notna(row['title']) and str(row['title']) != 'nan':
        title = str(row['title']).lower()
    else:
        title = ""
    
    if title:
        parts.extend([title] * 2)
    
    # Publisher crucial for textbooks
    if pd.notna(row.get('publisher')) and str(row.get('publisher')) != 'nan':
        publisher = str(row.get('publisher')).lower()
        parts.append(publisher)
    
    # Description from Google Books
    if pd.notna(row.get('description')) and str(row.get('description')) != 'nan':
        desc = str(row.get('description'))[:800].lower()
        parts.append(desc)
    
    # Subtitle adds context
    if pd.notna(row.get('subtitle')) and str(row.get('subtitle')) != 'nan':
        subtitle = str(row.get('subtitle')).lower()
        parts.append(subtitle)
    
    # Australian curriculum context
    if pd.notna(row['State']):
        state_terms = f"australia australian {str(row['State']).lower()}"
        parts.append(state_terms)
    
    # Add Trove corpus if available
    if pd.notna(row.get('corpus')) and str(row.get('corpus')) != 'nan':
        trove_corpus = str(row.get('corpus')).lower()
        parts.append(trove_corpus)
    
    # Add authors information
    if pd.notna(row.get('authors')) and str(row.get('authors')) != 'nan':
        authors = str(row.get('authors')).lower()
        parts.append(authors)
    elif pd.notna(row.get('author')) and str(row.get('author')) != 'nan':
        author = str(row.get('author')).lower()
        parts.append(author)
    
    return ' '.join(parts)

def calculate_quality_scores(df):
    """
    Calculate quality scores for books
    
    Args:
        df: DataFrame with book data
        
    Returns:
        Series with quality scores
    """
    def score_book(row):
        score = 0.5
        
        # Recent publications preferred
        if pd.notna(row.get('published_date')) and str(row.get('published_date')) != 'nan':
            try:
                pub_year = int(str(row.get('published_date'))[:4])
                if pub_year >= 2015:
                    score += 0.2
                elif pub_year >= 2010:
                    score += 0.1
            except:
                pass
        elif pd.notna(row.get('start_year')) and str(row.get('start_year')) != 'nan':
            try:
                pub_year = int(float(row.get('start_year')))
                if pub_year >= 2015:
                    score += 0.2
                elif pub_year >= 2010:
                    score += 0.1
            except:
                pass
        
        # Google Books ratings
        if pd.notna(row.get('average_rating')) and pd.notna(row.get('ratings_count')) and str(row.get('average_rating')) != 'nan' and str(row.get('ratings_count')) != 'nan':
            try:
                rating = float(row.get('average_rating'))
                count = float(row.get('ratings_count'))
                if count >= 10:
                    score += (rating - 3.0) / 10
            except:
                pass
        
        # Page count for textbooks
        if pd.notna(row.get('page_count')) and str(row.get('page_count')) != 'nan':
            try:
                pages = float(row.get('page_count'))
                if 150 <= pages <= 600:
                    score += 0.1
            except:
                pass
        
        # Rich description bonus
        if pd.notna(row.get('description')) and str(row.get('description')) != 'nan' and len(str(row.get('description'))) > 100:
            score += 0.1
        
        # Trove relevance score
        if pd.notna(row.get('relevance_score')) and str(row.get('relevance_score')) != 'nan':
            try:
                relevance = float(row.get('relevance_score'))
                score += min(0.1, relevance / 10)
            except:
                pass
        
        # Recency and popularity scores from Trove
        if pd.notna(row.get('recency_score')) and str(row.get('recency_score')) != 'nan':
            try:
                recency = float(row.get('recency_score'))
                score += recency * 0.05
            except:
                pass
        
        if pd.notna(row.get('popularity_score')) and str(row.get('popularity_score')) != 'nan':
            try:
                popularity = float(row.get('popularity_score'))
                score += popularity * 0.05
            except:
                pass
        
        return max(0, min(1, score))
    
    return df.apply(score_book, axis=1)

# Only run this if the file is executed directly
if __name__ == "__main__":
    # Load, process, and save
    input_file = os.path.join(data_dir, 'common_books.csv')
    output_file = os.path.join(data_dir, 'engineered_data.csv')

    df = pd.read_csv(input_file)
    df_engineered = engineer_educational_features(df)
    df_engineered.to_csv(output_file, index=False)

    print(f"Engineered dataset saved to '{output_file}'")
    print(f"Shape: {df_engineered.shape}")
    print(f"New columns: enhanced_corpus, subject_keywords, grade_category, quality_score, final_corpus")