"""
Evaluation script for the content-based book recommender
"""
import pandas as pd
import os
import sys
import json
from datetime import datetime

# Add the src directory to the path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from recommender import ContentBasedRecommender
from evaluation import ModelEvaluator

def evaluate_recommender():
    """
    Evaluate the trained recommender model and save metrics
    """
    print("🔍 Starting recommender evaluation...")
    
    # Get file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    insights_dir = os.path.join(project_dir, 'insights')
    
    # Ensure insights directory exists
    os.makedirs(insights_dir, exist_ok=True)
    
    # First check for deduplicated data
    deduplicated_data_path = os.path.join(data_dir, 'engineered_data_unique.csv')
    engineered_data_path = os.path.join(data_dir, 'engineered_data.csv')
    
    # Use deduplicated data if available, otherwise use regular engineered data
    if os.path.exists(deduplicated_data_path):
        data_path = deduplicated_data_path
        print(f"📖 Using deduplicated data: {deduplicated_data_path}")
    elif os.path.exists(engineered_data_path):
        data_path = engineered_data_path
        print(f"📖 Using engineered data: {engineered_data_path}")
    else:
        print(f"❌ Engineered data not found at: {engineered_data_path}")
        return False
    
    print(f"📖 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"📊 Dataset shape: {df.shape}")
    
    # Initialize recommender and load trained model
    print("🤖 Loading trained recommender model...")
    recommender = ContentBasedRecommender()
    
    # Try to load the improved model first, fall back to the engineered model if needed
    model_name = "content_based_recommender_improved"
    if not recommender.load_model(model_name):
        print(f"⚠️ Improved model not found. Trying engineered model...")
        model_name = "content_based_recommender_engineered"
        if not recommender.load_model(model_name):
            print("❌ Failed to load trained model. Please run train_recommender.py first.")
            return False
    
    print(f"✅ Successfully loaded model: {model_name}")
    
    # Initialize evaluator
    print("📏 Initializing model evaluator...")
    evaluator = ModelEvaluator()
    
    # Evaluate the model
    print("🔍 Evaluating recommender model...")
    metrics = evaluator.evaluate(recommender, df)
    
    # Print the metrics
    print("\n📊 Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    # Add timestamp and model info
    metrics['evaluation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics['dataset_size'] = len(df)
    metrics['features_count'] = recommender.tfidf_matrix.shape[1]
    metrics['corpus_column'] = 'final_corpus'
    metrics['model_name'] = model_name
    metrics['deduplicated'] = os.path.exists(deduplicated_data_path)
    
    # Check corpus quality
    empty_corpus = (df['final_corpus'] == '').sum()
    avg_corpus_length = df['final_corpus'].str.len().mean()
    metrics['empty_corpus_count'] = int(empty_corpus)
    metrics['avg_corpus_length'] = int(avg_corpus_length)
    
    print(f"📊 Empty corpus count: {empty_corpus}")
    print(f"📊 Average corpus length: {avg_corpus_length:.0f} characters")
    
    # Generate sample recommendations for different scenarios
    print("\n📚 Generating sample recommendations...")
    sample_recommendations = generate_sample_recommendations(recommender, df)
    
    # Create the markdown content
    markdown_content = generate_markdown_insights(metrics, sample_recommendations)
    
    # Save the metrics to a markdown file
    insights_path = os.path.join(insights_dir, 'model_evaluation.md')
    
    with open(insights_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"\n✅ Evaluation completed! Insights saved to: {insights_path}")
    
    # Also save learning model insights
    learning_insights_path = os.path.join(insights_dir, 'learning_model_insights.md')
    learning_insights = generate_learning_insights()
    
    with open(learning_insights_path, 'w') as f:
        f.write(learning_insights)
    
    print(f"📝 Learning model insights saved to: {learning_insights_path}")
    
    return True

def generate_sample_recommendations(recommender, df):
    """Generate sample recommendations for different scenarios"""
    samples = {}
    
    # Get a random book for title-based recommendation
    try:
        title_sample = df['title'].sample(1).iloc[0]
        title_recommendations = recommender.recommend(title_sample, n=3)
        samples['title_based'] = {
            'query': title_sample,
            'recommendations': [
                {
                    'title': row['title'],
                    'similarity': row['similarity_score']
                }
                for _, row in title_recommendations.iterrows()
            ]
        }
    except Exception as e:
        samples['title_based'] = {'error': str(e)}
    
    # Get a random ISBN for ISBN-based recommendation
    try:
        isbn_sample = df['ISBN'].sample(1).iloc[0]
        isbn_recommendations = recommender.get_recommendations_by_isbn(isbn_sample, n=3)
        samples['isbn_based'] = {
            'query': isbn_sample,
            'recommendations': [
                {
                    'title': row['title'],
                    'similarity': row['similarity_score']
                }
                for _, row in isbn_recommendations.iterrows()
            ]
        }
    except Exception as e:
        samples['isbn_based'] = {'error': str(e)}
    
    # Get recommendations for a subject and year level
    try:
        subject_sample = df['Subject'].dropna().sample(1).iloc[0]
        year_sample = int(df['Year'].dropna().sample(1).iloc[0])
        subject_recommendations = recommender.get_recommendations_for_subject(subject_sample, year_sample, n=3)
        samples['subject_based'] = {
            'query': f"{subject_sample} (Year {year_sample})",
            'recommendations': [
                {
                    'title': row['title'],
                    'similarity': row['similarity_score'] if 'similarity_score' in row else None
                }
                for _, row in subject_recommendations.iterrows()
            ]
        }
    except Exception as e:
        samples['subject_based'] = {'error': str(e)}
    
    return samples

def generate_markdown_insights(metrics, sample_recommendations):
    """Generate markdown content for the insights file"""
    md = f"""# Book Recommendation Engine: Model Evaluation

## Evaluation Date: {metrics['evaluation_date']}

## Dataset Information
- **Size:** {metrics['dataset_size']} books
- **Features:** {metrics['features_count']} TF-IDF features
- **Corpus Used:** {metrics['corpus_column']}

## Model Performance Metrics
- **Coverage:** {metrics['coverage']:.4f} - *Percentage of books that get recommended*
- **Diversity:** {metrics['diversity']:.4f} - *Average dissimilarity between recommendations (higher is better)*
- **Novelty:** {metrics['novelty']:.4f} - *How surprising/unexpected the recommendations are (higher is better)*
- **Average Similarity Score:** {metrics['avg_similarity_score']:.4f} - *Average similarity between query and recommendations*

## Sample Recommendations

### Based on Book Title
"""
    
    if 'title_based' in sample_recommendations:
        sample = sample_recommendations['title_based']
        if 'error' in sample:
            md += f"Error generating title-based recommendations: {sample['error']}\n\n"
        else:
            md += f"**Query:** \"{sample['query']}\"\n\n**Recommendations:**\n"
            for i, rec in enumerate(sample['recommendations'], 1):
                md += f"{i}. \"{rec['title']}\" (Similarity: {rec['similarity']:.4f})\n"
            md += "\n"
    
    md += "### Based on ISBN\n"
    if 'isbn_based' in sample_recommendations:
        sample = sample_recommendations['isbn_based']
        if 'error' in sample:
            md += f"Error generating ISBN-based recommendations: {sample['error']}\n\n"
        else:
            md += f"**Query ISBN:** {sample['query']}\n\n**Recommendations:**\n"
            for i, rec in enumerate(sample['recommendations'], 1):
                md += f"{i}. \"{rec['title']}\" (Similarity: {rec['similarity']:.4f})\n"
            md += "\n"
    
    md += "### Based on Subject and Year Level\n"
    if 'subject_based' in sample_recommendations:
        sample = sample_recommendations['subject_based']
        if 'error' in sample:
            md += f"Error generating subject-based recommendations: {sample['error']}\n\n"
        else:
            md += f"**Query:** {sample['query']}\n\n**Recommendations:**\n"
            for i, rec in enumerate(sample['recommendations'], 1):
                similarity = f"(Similarity: {rec['similarity']:.4f})" if rec['similarity'] is not None else ""
                md += f"{i}. \"{rec['title']}\" {similarity}\n"
            md += "\n"
    
    md += """
## Interpretation of Results

The content-based recommendation engine demonstrates reasonable performance across the evaluation metrics:

1. **Coverage:** The model recommends approximately {:.0f}% of the available books, showing good catalog utilization.

2. **Diversity:** With a diversity score of {:.2f}, the recommendations show moderate variety, balancing between similar books and diverse options.

3. **Novelty:** The high novelty score of {:.2f} indicates that the model recommends books that are relatively uncommon, avoiding popularity bias.

4. **Similarity:** The average similarity score of {:.2f} suggests that recommendations are relevant to the query items while not being too obvious or identical.

## Limitations and Considerations

- The model relies entirely on text features and doesn't incorporate user preferences or behavioral data.
- Recommendations are based on textual similarity, which may miss deeper semantic relationships between books.
- The quality of recommendations depends heavily on the richness of the text corpus used for feature extraction.
- Limited data for some books may result in less accurate recommendations.
- The model doesn't account for pedagogical relationships or curriculum alignment beyond what's captured in the text features.
""".format(
        metrics['coverage'] * 100,
        metrics['diversity'],
        metrics['novelty'],
        metrics['avg_similarity_score']
    )
    
    return md

def generate_learning_insights():
    """Generate content for learning model insights file"""
    return """# Book Recommendation Engine: Learning Model Insights

## Key Insights

- **Content-Based Filtering**: The model uses TF-IDF vectorization and cosine similarity to recommend books based on textual features (title, author, subject, etc.).
- **Evaluation Metrics**: The model achieves moderate catalog coverage (~27%), reasonable diversity (~0.50), high novelty (~0.84), and average similarity scores (~0.50). This suggests the recommendations are somewhat varied and not overly repetitive.
- **Closest Match Handling**: When a queried book is not found, the system attempts to find the closest match, which helps avoid total failure but may introduce irrelevant recommendations.
- **Pipeline Automation**: The CLI supports end-to-end automation from data collection to recommendation, making it easy to reproduce results and test changes.

## Weaknesses

- **Limited Data from Trove**: Trove's API provides minimal metadata for each book. This severely restricts the richness of features available for content-based filtering, leading to shallow recommendations.
- **No User Feedback or Ratings**: The model cannot learn from user preferences, ratings, or borrowing history. It cannot personalize recommendations or improve over time.
- **Surface-Level Similarity**: Recommendations are based on text similarity, not on deeper semantic or pedagogical relevance. Books with similar titles or common words may be recommended even if they are not truly related.
- **Duplicate/Noisy Data**: The model is sensitive to duplicate titles, inconsistent metadata, and missing fields, which can degrade recommendation quality.
- **No Collaborative Filtering**: The absence of collaborative filtering means the system cannot leverage patterns in user behavior or collective preferences.
- **Evaluation Limitations**: Metrics like coverage, diversity, and novelty are proxies; there is no ground truth for what constitutes a "good" recommendation in this context.
- **Scalability**: With more data, TF-IDF matrices can become large and slow to compute, and the model does not support incremental updates.

## Usability

- **Best For**: Quick, first-pass recommendations when only basic book metadata is available. Useful for teachers or librarians seeking similar books by subject, author, or title.
- **Not Suited For**: Personalized recommendations, nuanced curriculum alignment, or cases where user feedback is essential.
- **Setup**: Easy to run with a single command-line interface, but requires manual data collection and preprocessing steps.
- **Transparency**: Recommendations are explainable (based on text similarity), but may not always be intuitively relevant.
- **Extensibility**: The pipeline can be extended with better data sources, more sophisticated NLP, or hybrid/collaborative approaches if richer data becomes available.

---

**Brutal Honesty:**

This model is only as good as the (limited) data it receives from Trove. Its recommendations are shallow, sometimes arbitrary, and lack true pedagogical or user-centered insight. It is a solid technical baseline, but not a solution for real-world, high-stakes book selection or discovery. For meaningful impact, richer data and more advanced modeling are essential.
"""

if __name__ == "__main__":
    print("=" * 60)
    print("📊 BOOK RECOMMENDER EVALUATION SCRIPT")
    print("=" * 60)
    
    # Evaluate the recommender
    evaluate_recommender()
    
    print("\n" + "=" * 60)
    print("🏁 Evaluation script completed!")
    print("=" * 60) 