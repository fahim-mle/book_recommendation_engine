# MA5851 Assessment 1: NLP Recommendation Engine - Complete Work Pipeline

## Introduction

This project developed a book recommendation engine for Australian school teachers using NLP techniques. Beginning with limited textbook data (School_ID, State, Year, Subject, ISBN), we augmented it using the Trove API, then further enriched it with Google Books API. We engineered features for the corpus, built and improved a content-based recommendation system, created visualizations, and evaluated the model's performance.

## Task 1: Data Generation & Augmentation (20% of grade)

### API Integration

Our initial dataset contained only basic information about textbooks used in Australian schools. To build an effective recommendation system, we needed to enrich this data with descriptions, ratings, and other metadata.

**Multi-API Approach:**
1. **Trove API**: We first implemented data collection using the Trove API to gather basic book information from Australia's national library database. This provided us with titles, authors, and publishing years but lacked rich descriptions.

2. **Google Books API**: To overcome Trove's limitations, we implemented a second data collection module using the Google Books API, which provided fuller descriptions, ratings, page counts, and categories.

Both APIs required thoughtful integration:

```python
# Our API client implementation included rate limiting and error handling
def fetch_book_data(isbn):
    # Add delay between requests to respect rate limits
    time.sleep(0.5)
    try:
        response = requests.get(f"{API_BASE_URL}?q=isbn:{isbn}&key={API_KEY}")
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Error fetching ISBN {isbn}: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Exception for ISBN {isbn}: {str(e)}")
        return None
```

The code included robust error handling, logging failed requests to a dedicated log file (`failed_isbns.log`), and implementing absolute paths to ensure the script could be run from any directory in the project.

### Data Augmentation

Once data was collected, we merged it with our original dataset:

1. We created `augmented_data.csv` containing basic metadata from Trove
2. We then enhanced this with Google Books data, adding descriptions, ratings, etc.
3. The final merged dataset formed the foundation for our recommendation engine

This multi-API approach significantly improved data quality, addressing a key challenge in educational recommendation systems: the lack of rich textbook metadata.

## Task 2: Data Wrangling & EDA (30% of grade)

### Corpus Data Wrangling

Text preprocessing was critical for effective recommendations:

1. **Initial Corpus Creation**: Combined title, author, subject, and description fields
2. **Enhanced Corpus**: Created a weighted corpus giving priority to educational fields:
   - Subject: 3x weight (reduced from initial 6x)
   - Year level: 2x weight
   - Title: 2x weight
   - Description: 1x weight

3. **Deduplication**: A critical step where we identified and removed 2,032 duplicate ISBN entries, reducing the dataset from 3,103 to 1,071 unique books. This significantly improved model performance.

4. **Feature Engineering**:
   - Extracted subject keywords
   - Created grade categories (early_primary, primary, junior_secondary, senior_secondary)
   - Calculated quality scores based on ratings, recency, and description richness

### Exploratory Data Analysis

Our EDA revealed important insights:

- **Distribution Analysis**: Visualized subject, year level, and state distributions to understand the dataset composition
- **Quality Metrics**: Analyzed the distribution of quality scores
- **Corpus Analysis**: Examined corpus length and its relationship to recommendation quality
- **Wordclouds**: Created wordclouds for the overall corpus and by subject

Visualizations stored in the `visualization` directory showed that:
- English and Mathematics dominated the subject distribution
- Years 10-12 had the highest representation
- Victoria and NSW had the most books
- Quality scores followed a normal distribution with a mean around 0.6

### Data Splitting

For model evaluation:
- 80% training / 20% testing split
- Stratified by subject to maintain distribution
- Special attention to ensure sufficient representation across year levels

## Task 3: Content-Based NLP Recommender (40% of grade)

### Feature Engineering

We implemented advanced text vectorization:

```python
# Improved TF-IDF settings
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,      # Reduced from initial 15000
    max_df=0.7,             # More restrictive
    min_df=1,               # Keep more terms
    ngram_range=(1,2)       # Include bigrams
)
```

This approach:
1. Removed common stop words
2. Focused on the most relevant 5,000 features
3. Ignored terms appearing in >70% of documents
4. Included bigrams to capture phrases like "year level" or "primary school"

### Machine Learning Implementation

Our content-based filtering approach:

1. **TF-IDF Vectorization**: Transformed the engineered corpus into a numerical representation
2. **Cosine Similarity**: Calculated similarity between book vectors
3. **Minimum Similarity Threshold**: Filtered recommendations below 0.1 similarity

We integrated the model into a robust recommender class that supported:
- Title-based recommendations
- ISBN-based recommendations
- Subject and year level recommendations
- Model persistence (saving/loading)

### Evaluation

Performance metrics showed significant improvement after our optimizations:

| Metric | Initial Model | Optimized Model |
|--------|--------------|----------------|
| Coverage | 5.76% | 29.41% |
| Diversity | 0.37 | 0.45 |
| Novelty | 0.93 | 0.77 |
| Avg. Similarity | 0.63 | 0.55 |

The insights document (`insights/model_evaluation.md`) provided a detailed analysis of these metrics, explaining how each contributes to recommendation quality.

### Demonstration

We created an interactive CLI interface (`use_recommender.py`) allowing users to:
1. Get recommendations by book title
2. Get recommendations by ISBN
3. Get recommendations for subject and year level
4. Get random book recommendations

Example recommendations for "Macbeth":
```
1. "Macbeth / William Shakespeare (ISBN: 9781586638467)"
   Subject: ENGLISH, Year: 10
   Similarity: 0.900

2. "The Merchant of Venice / William Shakespeare"
   Subject: ENGLISH, Year: 10
   Similarity: 0.833
```

## Task 4: Integration & Challenge Analysis (10% of grade)

### Integration Plan

The recommendation system is designed for Australian secondary schools (Years 7-12):

**Deployment Strategy:**
1. **Web Application**: Convert the CLI to a web interface accessible to teachers
2. **School LMS Integration**: API endpoints for integration with learning management systems
3. **Offline Mode**: Allow downloading recommendations for use without internet

**Workflow Integration:**
1. Teachers search by curriculum topic or existing textbook
2. System suggests alternatives and complementary resources
3. Recommendations prioritize resources available in school library or digital subscriptions

### Challenge Identification

The primary challenge is data quality and consistency:

1. **Inconsistent Metadata**: Many books lack complete descriptions or subject classifications
2. **Limited Australian Context**: Google Books provides less detailed information for Australian curriculum materials
3. **Cold Start Problem**: New textbooks have no usage history

### Solution Proposal

1. **Enhanced Data Collection**:
   - Partner with Australian publishers for direct metadata feeds
   - Implement teacher feedback loop to improve data quality
   - Collaborate with state education departments for curriculum alignment

2. **Hybrid Approach**:
   - Combine content-based filtering with rule-based systems
   - Integrate explicit curriculum mapping when available
   - Add collaborative filtering as usage data grows

3. **Resource Requirements**:
   - Cloud infrastructure for scaling (AWS/Azure)
   - Data engineering team for ongoing data enrichment
   - UX designers for teacher-friendly interface

## Final Assessment

The book recommendation engine demonstrates the power of NLP in educational resource discovery. Through iterative improvements, we addressed critical challenges:

1. **Data Quality**: Through multi-API integration and deduplication
2. **Feature Engineering**: By creating an optimized corpus with educational context
3. **Model Performance**: Increasing coverage from 5.76% to 29.41%

The system meets the core requirements of providing relevant, diverse recommendations for teachers while acknowledging limitations of a purely content-based approach.

## Grade Assessment

Based on the project requirements and outcomes:

| Component | Weight | Self-Assessment | Justification |
|-----------|--------|-----------------|---------------|
| Data Generation | 20% | High Distinction | Implemented multiple APIs with robust error handling |
| Data Wrangling & EDA | 30% | Distinction | Thorough data cleaning, visualization, and feature engineering |
| NLP Recommender | 40% | Credit/Distinction | Effective model with iterative improvements, but limitations in evaluation |
| Integration Analysis | 10% | Credit | Realistic integration plan but could be more detailed |

The project demonstrates a comprehensive understanding of NLP recommendation systems, from data collection to model optimization, with meaningful evaluation and visualization throughout the pipeline. 