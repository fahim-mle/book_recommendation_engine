# Book Recommendation Engine

A content-based book recommendation system for Australian schools using natural language processing (NLP) techniques to analyze textbook metadata and provide relevant recommendations.

## Project Overview

This project implements a book recommendation engine for educational contexts. The system:
- Collects book data from both the Trove API and Google Books API
- Combines and processes the data using NLP techniques
- Applies content-based filtering to generate recommendations
- Provides an interactive CLI for easy usage

The recommendation engine is specifically tailored for the Australian education context, with features focusing on subjects, year levels, and curriculum relevance.

## Key Features

- **Dual API Data Collection**: Retrieves book metadata from Trove API and Google Books API
- **Advanced Text Processing**: Creates enhanced corpus with weighted educational features
- **Content-Based Filtering**: Uses TF-IDF vectorization and cosine similarity
- **Educational Context**: Considers subject, year level, and curriculum relevance
- **Deduplication**: Removes duplicate ISBNs to improve model performance
- **Interactive CLI**: Easy-to-use interface for generating recommendations
- **Data Visualization**: Provides insights into the book dataset
- **Performance Metrics**: Evaluates the recommender using coverage, diversity, novelty, and similarity

## Project Structure

```
book_recommendation_engine/
├── data/                       # Data directory
│   ├── initial_data.csv        # Initial ISBN dataset
│   ├── augmented_data.csv      # Data enriched with Trove API
│   ├── google_book_data.csv    # Data from Google Books API
│   ├── common_books.csv        # Combined data from both APIs
│   ├── engineered_data.csv     # Processed data with features
│   └── engineered_data_unique.csv # Deduplicated data
├── documentation/              # Project documentation
│   ├── how-to-run.md           # Step-by-step instructions
│   └── project_documentation.md # Comprehensive documentation
├── insights/                   # Evaluation results
│   ├── model_evaluation.md     # Performance metrics
│   └── learning_model_insights.md # Analysis of the model's strengths/weaknesses
├── notebooks/                  # Jupyter notebooks for demonstration
├── src/                        # Source code
│   ├── data_collection.py      # Trove API integration
│   ├── data_collection_google_api.py # Google Books API integration
│   ├── data_collection_v2.py   # Combines data from both APIs
│   ├── data_processing.py      # Text preprocessing & feature engineering
│   ├── data_visualization.py   # Creates visualizations
│   ├── deduplicate_data.py     # Removes duplicate ISBNs
│   ├── evaluate_recommender.py # Evaluates model performance
│   ├── recommender.py          # Core recommendation engine
│   ├── train_recommender.py    # Trains the recommendation model
│   ├── use_recommender.py      # Interactive CLI
│   └── main.py                 # Command-line interface
├── trained_models/             # Saved models
│   └── content_based_recommender_improved.joblib
├── visualization/              # Data visualizations
└── requirements.txt            # Project dependencies
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Trove API key (for data collection)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd book_recommendation_engine/completed_project
```

2. Create and activate a virtual environment:

#### For Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage Guide

### Step 1: Data Collection

Collect book data from Trove API and Google Books API:

```bash
# Collect data from Trove API (requires API key)
python src/main.py collect --api-key YOUR_API_KEY --input data/initial_data.csv --output data/augmented_data.csv

# Collect data from Google Books API
python src/main.py collect_with_google --input data/initial_data.csv

# Merge data from both APIs
python src/data_collection_v2.py
```

The first command collects data from the Trove API using your API key, while the second command collects data from the Google Books API (which doesn't require an API key). The third command merges the data from both APIs to create a combined dataset (`data/common_books.csv`).

### Step 2: Data Processing

Process and enhance the collected data:

```bash
# Process and engineer features
python src/main.py process --input data/common_books.csv --output data/engineered_data.csv

# Remove duplicate ISBNs
python src/deduplicate_data.py
```

The data processing step extracts educational features, creates an enhanced corpus with weighted fields, and calculates quality scores for each book. The deduplication step removes duplicate ISBNs to improve model performance.

### Step 3: Train the Recommendation Model

Train the content-based recommendation model:

```bash
python src/main.py train --input data/engineered_data_unique.csv
```

This trains the model using TF-IDF vectorization on the deduplicated data and saves the model to `trained_models/content_based_recommender_improved.joblib`.

### Step 4: Evaluate the Model

Evaluate the model's performance:

```bash
python src/evaluate_recommender.py
```

This generates evaluation metrics including coverage, diversity, novelty, and similarity scores, and saves the results to the `insights` directory.

### Step 5: Generate Recommendations

Use the interactive recommendation tool:

```bash
python src/use_recommender.py
```

This launches an interactive CLI with the following options:
1. Get recommendations by book title
2. Get recommendations by ISBN
3. Get recommendations for a specific subject and year level
4. Show random book recommendations

Alternatively, you can generate recommendations directly using:

```bash
# Get recommendations by title
python src/main.py recommend --data data/engineered_data_unique.csv --title "Macbeth" --n 5

# Get recommendations by ISBN
python src/main.py recommend --data data/engineered_data_unique.csv --isbn 9780141439518 --n 5
```

### Step 6: Generate Visualizations (Optional)

Create visualizations to better understand the data:

```bash
python src/data_visualization.py
```

This generates various visualizations including subject distribution, word clouds, and quality score distribution, and saves them to the `visualization` directory.

## Data Processing Details

The data processing pipeline includes:

1. **Corpus Creation**: Combines title, author, subject, and description
2. **Educational Feature Extraction**: 
   - Categorizes year levels (early primary, primary, junior secondary, senior secondary)
   - Extracts subject keywords for mathematics, science, English, etc.
3. **Enhanced Corpus Creation**: 
   - Applies weighted importance to different fields (subject gets 3x weight, year gets 2x weight, etc.)
   - Incorporates curriculum context, publisher information, and other metadata
4. **Quality Score Calculation**: 
   - Considers publication date, ratings, page count, and description quality
   - Provides a score between 0 and 1 for each book

## Model Performance

The current model achieves the following metrics:

- **Coverage**: 30.59% - Percentage of books that get recommended
- **Diversity**: 49.28% - Average dissimilarity between recommendations
- **Novelty**: 77.48% - How surprising/unexpected the recommendations are
- **Similarity**: 50.72% - Average similarity between query and recommendations

## Running the Complete Pipeline

To run the entire pipeline with a single command:

```bash
python src/main.py pipeline --api-key YOUR_API_KEY --input data/initial_data.csv
```

This will:
1. Collect data from Trove API
2. Process the data
3. Train the recommendation model
4. Generate recommendations (if --title or --isbn is provided)

## Troubleshooting

Common issues:

1. **Missing API key**: Make sure you have a valid Trove API key for data collection
2. **FileNotFoundError**: Verify all referenced data files exist before running scripts
3. **ModuleNotFoundError**: Make sure you've activated the virtual environment
4. **ImportError**: Ensure you're running the scripts from the project root directory
5. **NaN values in recommendations**: The system handles missing data, but recommendation quality improves with complete data

## License

See the LICENSE file for details. 