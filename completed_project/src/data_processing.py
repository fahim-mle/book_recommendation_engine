import pandas as pd
import numpy as np
import os

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # Go up one level to completed_project
data_dir = os.path.join(project_dir, 'data')

def engineer_educational_features(df):
   """Enhanced feature engineering for educational textbook recommendations"""
   df_enhanced = df.copy()
   
   # Create weighted educational corpus
   def create_enhanced_corpus(row):
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
       if pd.notna(row['title_google']) and str(row['title_google']) != 'nan':
           title = str(row['title_google']).lower()
       elif pd.notna(row['title']) and str(row['title']) != 'nan':
           title = str(row['title']).lower()
       else:
           title = ""
       
       if title:
           parts.extend([title] * 2)
       
       # Publisher crucial for textbooks
       if pd.notna(row['publisher']) and str(row['publisher']) != 'nan':
           publisher = str(row['publisher']).lower()
           parts.append(publisher)
       
       # Description from Google Books
       if pd.notna(row['description']) and str(row['description']) != 'nan':
           desc = str(row['description'])[:800].lower()
           parts.append(desc)
       
       # Subtitle adds context
       if pd.notna(row['subtitle']) and str(row['subtitle']) != 'nan':
           subtitle = str(row['subtitle']).lower()
           parts.append(subtitle)
       
       # Australian curriculum context
       if pd.notna(row['State']):
           state_terms = f"australia australian {str(row['State']).lower()}"
           parts.append(state_terms)
       
       # Add Trove corpus if available
       if pd.notna(row['corpus']) and str(row['corpus']) != 'nan':
           trove_corpus = str(row['corpus']).lower()
           parts.append(trove_corpus)
       
       # Add authors information
       if pd.notna(row['authors']) and str(row['authors']) != 'nan':
           authors = str(row['authors']).lower()
           parts.append(authors)
       elif pd.notna(row['author']) and str(row['author']) != 'nan':
           author = str(row['author']).lower()
           parts.append(author)
       
       return ' '.join(parts)
   
   df_enhanced['enhanced_corpus'] = df_enhanced.apply(create_enhanced_corpus, axis=1)
   
   # Add subject-specific keywords
   def add_subject_keywords(subject):
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
   
   df_enhanced['subject_keywords'] = df_enhanced['Subject'].apply(add_subject_keywords)
   
   # Grade level categorization
   def categorize_grade_level(year):
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
   
   df_enhanced['grade_category'] = df_enhanced['Year'].apply(categorize_grade_level)
   
   # Quality scoring
   def calculate_educational_quality(row):
       score = 0.5
       
       # Recent publications preferred
       if pd.notna(row['published_date']) and str(row['published_date']) != 'nan':
           try:
               pub_year = int(str(row['published_date'])[:4])
               if pub_year >= 2015:
                   score += 0.2
               elif pub_year >= 2010:
                   score += 0.1
           except:
               pass
       elif pd.notna(row['start_year']) and str(row['start_year']) != 'nan':
           try:
               pub_year = int(float(row['start_year']))
               if pub_year >= 2015:
                   score += 0.2
               elif pub_year >= 2010:
                   score += 0.1
           except:
               pass
       
       # Google Books ratings
       if pd.notna(row['average_rating']) and pd.notna(row['ratings_count']) and str(row['average_rating']) != 'nan' and str(row['ratings_count']) != 'nan':
           try:
               rating = float(row['average_rating'])
               count = float(row['ratings_count'])
               if count >= 10:
                   score += (rating - 3.0) / 10
           except:
               pass
       
       # Page count for textbooks
       if pd.notna(row['page_count']) and str(row['page_count']) != 'nan':
           try:
               pages = float(row['page_count'])
               if 150 <= pages <= 600:
                   score += 0.1
           except:
               pass
       
       # Rich description bonus
       if pd.notna(row['description']) and str(row['description']) != 'nan' and len(str(row['description'])) > 100:
           score += 0.1
       
       # Trove relevance score
       if pd.notna(row['relevance_score']) and str(row['relevance_score']) != 'nan':
           try:
               relevance = float(row['relevance_score'])
               score += min(0.1, relevance / 10)
           except:
               pass
       
       # Recency and popularity scores from Trove
       if pd.notna(row['recency_score']) and str(row['recency_score']) != 'nan':
           try:
               recency = float(row['recency_score'])
               score += recency * 0.05
           except:
               pass
       
       if pd.notna(row['popularity_score']) and str(row['popularity_score']) != 'nan':
           try:
               popularity = float(row['popularity_score'])
               score += popularity * 0.05
           except:
               pass
       
       return max(0, min(1, score))
   
   df_enhanced['quality_score'] = df_enhanced.apply(calculate_educational_quality, axis=1)
   
   # Final comprehensive corpus
   df_enhanced['final_corpus'] = (
       df_enhanced['enhanced_corpus'] + ' ' +
       df_enhanced['subject_keywords'] + ' ' +
       df_enhanced['grade_category']
   )
   
   return df_enhanced

# Load, process, and save
input_file = os.path.join(data_dir, 'common_books.csv')
output_file = os.path.join(data_dir, 'engineered_data.csv')

df = pd.read_csv(input_file)
df_engineered = engineer_educational_features(df)
df_engineered.to_csv(output_file, index=False)

print(f"Engineered dataset saved to '{output_file}'")
print(f"Shape: {df_engineered.shape}")
print(f"New columns: enhanced_corpus, subject_keywords, grade_category, quality_score, final_corpus")