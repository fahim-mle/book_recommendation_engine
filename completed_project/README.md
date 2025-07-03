# Book Recommendation Engine

A content-based book recommendation system for teachers in Australian schools based on textbook lists.

## Overview

This project implements a book recommendation engine using Natural Language Processing (NLP) techniques. The system augments book data from the Trove API and builds a content-based filtering recommendation system.

## Features

- Data collection from Trove API
- Text preprocessing and feature engineering
- Content-based recommendation using TF-IDF and cosine similarity
- Evaluation metrics for recommendation quality
- Command-line interface for all operations

## Project Structure

```
book_recommendation_engine/
├── data/                # Data directory
├── models/              # Saved models (if implemented)
├── notebooks/           # Jupyter notebooks
└── src/                 # Source code
    ├── __init__.py
    ├── data_collection.py  # API integration
    ├── data_processing.py  # Text preprocessing
    ├── recommender.py      # Recommendation engine
    ├── evaluation.py       # Model evaluation
    └── main.py             # Main script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book_recommendation_engine.git
cd book_recommendation_engine
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection

Collect data from the Trove API:

```bash
python -m src.main collect --api-key YOUR_API_KEY --input data/initial_data.csv --output data/augmented_data.csv
```

### Data Processing

Process the collected data:

```bash
python -m src.main process --input data/augmented_data.csv --output data/processed_data.csv
```

### Training the Recommender

Train the recommendation engine:

```bash
python -m src.main train --input data/processed_data.csv
```

### Generating Recommendations

Generate recommendations based on a book title:

```bash
python -m src.main recommend --data data/processed_data.csv --title "Book Title"
```

Or based on ISBN:

```bash
python -m src.main recommend --data data/processed_data.csv --isbn 9781234567890
```

### Running the Complete Pipeline

Run the entire pipeline from data collection to recommendations:

```bash
python -m src.main pipeline --api-key YOUR_API_KEY --input data/initial_data.csv --title "Book Title"
```

## Data Sources

- Initial dataset: School textbook ISBNs
- Augmented data: [Trove API](https://trove.nla.gov.au/about/create-something/using-api)

## License

See the LICENSE file for details. 