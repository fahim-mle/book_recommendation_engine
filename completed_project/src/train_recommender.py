"""
Training script for the content-based book recommender
"""
import pandas as pd
import os
import sys

# Add the src directory to the path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from recommender import ContentBasedRecommender

def train_and_save_recommender():
    """
    Train the recommender on engineered data and save the model
    """
    print("🚀 Starting recommender training...")
    
    # Get file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    # Load engineered data
    engineered_data_path = os.path.join(data_dir, 'engineered_data.csv')
    
    if not os.path.exists(engineered_data_path):
        print(f"❌ Engineered data not found at: {engineered_data_path}")
        print("Please run data_processing.py first to create engineered_data.csv")
        return False
    
    print(f"📖 Loading engineered data from: {engineered_data_path}")
    df = pd.read_csv(engineered_data_path)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Available columns: {list(df.columns)}")
    
    # Check if we have the required columns
    required_columns = ['final_corpus', 'title', 'ISBN']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Initialize and train the recommender
    print("🤖 Initializing recommender...")
    recommender = ContentBasedRecommender()
    
    print("🎯 Training recommender on engineered data...")
    recommender.fit(df)
    
    # Save the trained model
    print("💾 Saving trained model...")
    model_path = recommender.save_model("content_based_recommender_engineered")
    
    print(f"✅ Training completed successfully!")
    print(f"📁 Model saved to: {model_path}")
    
    # Test the model with a sample recommendation
    print("\n🧪 Testing the trained model...")
    try:
        # Get a sample book title
        sample_title = df['title'].iloc[0]
        print(f"📚 Sample book: {sample_title}")
        
        # Get recommendations
        recommendations = recommender.recommend(sample_title, n=3)
        
        print(f"🎯 Top 3 recommendations for '{sample_title}':")
        for i, (_, book) in enumerate(recommendations.iterrows(), 1):
            print(f"  {i}. {book['title']} (Similarity: {book['similarity_score']:.3f})")
            
    except Exception as e:
        print(f"⚠️  Test recommendation failed: {e}")
    
    return True

def load_and_test_model():
    """
    Load a saved model and test it
    """
    print("\n🔄 Testing model loading...")
    
    recommender = ContentBasedRecommender()
    
    if recommender.load_model("content_based_recommender_engineered"):
        print("✅ Model loaded successfully!")
        
        # Test with a sample recommendation
        try:
            # Get a random book title from the loaded data
            sample_title = recommender.books_df['title'].sample(1).iloc[0]
            print(f"📚 Testing with book: {sample_title}")
            
            recommendations = recommender.recommend(sample_title, n=3)
            
            print(f"🎯 Top 3 recommendations for '{sample_title}':")
            for i, (_, book) in enumerate(recommendations.iterrows(), 1):
                print(f"  {i}. {book['title']} (Similarity: {book['similarity_score']:.3f})")
                
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
    else:
        print("❌ Failed to load model")

if __name__ == "__main__":
    print("=" * 60)
    print("📚 BOOK RECOMMENDER TRAINING SCRIPT")
    print("=" * 60)
    
    # Train and save the model
    success = train_and_save_recommender()
    
    if success:
        # Test loading the model
        load_and_test_model()
    
    print("\n" + "=" * 60)
    print("🏁 Training script completed!")
    print("=" * 60) 