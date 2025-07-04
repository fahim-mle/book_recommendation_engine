"""
Data Visualization Module - Creates insightful visualizations for the book recommendation engine
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

class DataVisualizer:
    """
    Class for creating visualizations from the engineered book data
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        # Get file paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.dirname(self.script_dir)
        self.data_dir = os.path.join(self.project_dir, 'data')
        self.visualization_dir = os.path.join(self.project_dir, 'visualization')
        
        # Create visualization directory if it doesn't exist
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Set the engineered data path
        self.engineered_data_path = os.path.join(self.data_dir, 'engineered_data.csv')
        
        # Load data
        self.df = None
        
    def load_data(self):
        """Load the engineered data"""
        print(f"📊 Loading engineered data from {self.engineered_data_path}")
        self.df = pd.read_csv(self.engineered_data_path)
        print(f"✅ Loaded data with shape: {self.df.shape}")
        return self.df
    
    def save_fig(self, fig, filename, format='png', dpi=300):
        """Save a matplotlib figure to the visualization directory"""
        filepath = os.path.join(self.visualization_dir, f"{filename}.{format}")
        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"📄 Saved figure to {filepath}")
    
    def save_plotly(self, fig, filename):
        """Save a plotly figure to the visualization directory"""
        filepath = os.path.join(self.visualization_dir, f"{filename}.html")
        fig.write_html(filepath)
        print(f"📄 Saved interactive plot to {filepath}")
        
        # Also save as image for report
        img_path = os.path.join(self.visualization_dir, f"{filename}.png")
        fig.write_image(img_path)
        print(f"📄 Saved static image to {img_path}")
    
    def create_subject_distribution(self):
        """Create visualization of subject distribution"""
        print("📊 Creating subject distribution visualization...")
        
        # Count subjects
        subject_counts = self.df['Subject'].value_counts().head(15)
        
        # Create plotly bar chart
        fig = px.bar(
            x=subject_counts.index,
            y=subject_counts.values,
            title="Top 15 Subjects in the Dataset",
            labels={'x': 'Subject', 'y': 'Number of Books'},
            color=subject_counts.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            width=1000,
            coloraxis_showscale=False
        )
        
        self.save_plotly(fig, "subject_distribution")
        
        return fig
    
    def create_year_distribution(self):
        """Create visualization of year level distribution"""
        print("📊 Creating year level distribution visualization...")
        
        # Count year levels
        year_counts = self.df['Year'].value_counts().sort_index()
        
        # Create plotly line chart
        fig = px.line(
            x=year_counts.index,
            y=year_counts.values,
            markers=True,
            title="Distribution of Books by Year Level",
            labels={'x': 'Year Level', 'y': 'Number of Books'}
        )
        
        # Add bar chart on the same figure
        fig.add_trace(
            go.Bar(
                x=year_counts.index,
                y=year_counts.values,
                opacity=0.5,
                name="Count"
            )
        )
        
        fig.update_layout(
            height=500,
            width=900,
            showlegend=False
        )
        
        self.save_plotly(fig, "year_distribution")
        
        return fig
    
    def create_state_distribution(self):
        """Create visualization of state distribution"""
        print("📊 Creating state distribution visualization...")
        
        # Count states
        state_counts = self.df['State'].value_counts()
        
        # Create plotly pie chart
        fig = px.pie(
            values=state_counts.values,
            names=state_counts.index,
            title="Distribution of Books by State",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        fig.update_layout(
            height=600,
            width=800
        )
        
        self.save_plotly(fig, "state_distribution")
        
        return fig
    
    def create_quality_score_distribution(self):
        """Create visualization of quality score distribution"""
        print("📊 Creating quality score distribution visualization...")
        
        # Create plotly histogram
        fig = px.histogram(
            self.df,
            x="quality_score",
            nbins=50,
            title="Distribution of Book Quality Scores",
            labels={'quality_score': 'Quality Score'},
            color_discrete_sequence=['#3366CC']
        )
        
        fig.update_layout(
            height=500,
            width=900,
            bargap=0.1
        )
        
        # Add a vertical line for the mean
        mean_quality = self.df['quality_score'].mean()
        fig.add_vline(x=mean_quality, line_dash="dash", line_color="red")
        
        # Check if we have y data and add annotation
        if fig.data and hasattr(fig.data[0], 'y') and fig.data[0].y is not None and len(fig.data[0].y) > 0:
            max_y = fig.data[0].y.max()
            fig.add_annotation(
                x=mean_quality,
                y=max_y * 0.95,
                text=f"Mean: {mean_quality:.3f}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-40
            )
        else:
            # Add annotation at a reasonable position if we can't calculate max y
            fig.add_annotation(
                x=mean_quality,
                y=10,  # arbitrary position
                text=f"Mean: {mean_quality:.3f}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-40
            )
        
        self.save_plotly(fig, "quality_score_distribution")
        
        return fig
    
    def create_subject_year_heatmap(self):
        """Create heatmap of subjects by year level"""
        print("📊 Creating subject-year heatmap...")
        
        # Create cross-tabulation of subjects and years
        top_subjects = self.df['Subject'].value_counts().head(10).index
        year_subject_df = self.df[self.df['Subject'].isin(top_subjects)]
        
        # Create pivot table
        pivot_table = pd.crosstab(year_subject_df['Year'], year_subject_df['Subject'])
        
        # Create plotly heatmap
        fig = px.imshow(
            pivot_table,
            labels=dict(x="Subject", y="Year Level", color="Number of Books"),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale="Viridis",
            title="Distribution of Top 10 Subjects Across Year Levels"
        )
        
        fig.update_layout(
            height=700,
            width=1000,
            xaxis_tickangle=-45
        )
        
        self.save_plotly(fig, "subject_year_heatmap")
        
        return fig
    
    def create_wordcloud(self, column='final_corpus', title='Word Cloud of Book Corpus'):
        """Create wordcloud from text data"""
        print(f"🔤 Creating wordcloud from {column}...")
        
        # Combine all text in the column
        text = ' '.join(self.df[column].dropna().astype(str))
        
        # Create stopwords set
        stopwords = set(STOPWORDS)
        stopwords.update(['isbn', 'book', 'books', 'edition', 'year', 'subject'])
        
        # Create wordcloud
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color='white',
            stopwords=stopwords,
            max_words=200,
            colormap='viridis',
            random_state=42
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=20)
        ax.axis('off')
        
        # Save figure
        filename = f"wordcloud_{column}"
        self.save_fig(fig, filename)
        
        return fig
    
    def create_subject_wordclouds(self):
        """Create wordclouds for top subjects"""
        print("🔤 Creating subject-specific wordclouds...")
        
        # Get top 5 subjects
        top_subjects = self.df['Subject'].value_counts().head(5).index
        
        for subject in top_subjects:
            print(f"  - Creating wordcloud for {subject}...")
            
            # Filter data for this subject
            subject_text = ' '.join(self.df[self.df['Subject'] == subject]['final_corpus'].dropna().astype(str))
            
            # Create stopwords set
            stopwords = set(STOPWORDS)
            stopwords.update(['isbn', 'book', 'books', 'edition', 'year', 'subject', subject.lower()])
            
            # Create wordcloud
            wordcloud = WordCloud(
                width=1000,
                height=600,
                background_color='white',
                stopwords=stopwords,
                max_words=100,
                colormap='plasma',
                random_state=42
            ).generate(subject_text)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f"Word Cloud for {subject}", fontsize=18)
            ax.axis('off')
            
            # Save figure
            filename = f"wordcloud_subject_{subject}"
            self.save_fig(fig, filename)
    
    def create_quality_by_subject_boxplot(self):
        """Create boxplot of quality scores by subject"""
        print("📊 Creating quality score by subject boxplot...")
        
        # Get top 10 subjects
        top_subjects = self.df['Subject'].value_counts().head(10).index
        subject_quality = self.df[self.df['Subject'].isin(top_subjects)]
        
        # Create plotly boxplot
        fig = px.box(
            subject_quality,
            x="Subject",
            y="quality_score",
            color="Subject",
            title="Quality Score Distribution by Subject",
            labels={'quality_score': 'Quality Score'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            height=600,
            width=1000,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        self.save_plotly(fig, "quality_by_subject_boxplot")
        
        return fig
    
    def create_corpus_length_analysis(self):
        """Analyze and visualize corpus length"""
        print("📊 Creating corpus length analysis...")
        
        # Calculate corpus length
        self.df['corpus_length'] = self.df['final_corpus'].fillna('').apply(len)
        
        # Create histogram
        fig = px.histogram(
            self.df,
            x="corpus_length",
            nbins=50,
            title="Distribution of Corpus Length",
            labels={'corpus_length': 'Corpus Length (characters)'},
            color_discrete_sequence=['#6633CC']
        )
        
        fig.update_layout(
            height=500,
            width=900,
            bargap=0.1
        )
        
        self.save_plotly(fig, "corpus_length_distribution")
        
        # Create scatter plot of corpus length vs quality score
        fig2 = px.scatter(
            self.df,
            x="corpus_length",
            y="quality_score",
            color="Subject",
            title="Relationship Between Corpus Length and Quality Score",
            labels={'corpus_length': 'Corpus Length (characters)', 'quality_score': 'Quality Score'},
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig2.update_layout(
            height=600,
            width=1000,
            showlegend=False
        )
        
        self.save_plotly(fig2, "corpus_length_vs_quality")
        
        return fig, fig2
    
    def create_year_level_analysis(self):
        """Create visualizations for year level analysis"""
        print("📊 Creating year level analysis...")
        
        # Calculate average quality score by year level
        year_quality = self.df.groupby('Year')['quality_score'].agg(['mean', 'count']).reset_index()
        
        # Create plotly bar chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add quality score line
        fig.add_trace(
            go.Scatter(
                x=year_quality['Year'],
                y=year_quality['mean'],
                mode='lines+markers',
                name='Avg. Quality Score',
                marker=dict(size=10)
            ),
            secondary_y=False
        )
        
        # Add book count bars
        fig.add_trace(
            go.Bar(
                x=year_quality['Year'],
                y=year_quality['count'],
                name='Number of Books',
                opacity=0.7
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title_text="Average Quality Score and Book Count by Year Level",
            height=600,
            width=1000
        )
        
        # Update axes
        fig.update_xaxes(title_text="Year Level")
        fig.update_yaxes(title_text="Average Quality Score", secondary_y=False)
        fig.update_yaxes(title_text="Number of Books", secondary_y=True)
        
        self.save_plotly(fig, "year_level_analysis")
        
        return fig
    
    def create_pca_visualization(self):
        """Create PCA visualization of book corpus"""
        print("📊 Creating PCA visualization of book corpus...")
        
        # Get top subjects for coloring
        top_subjects = self.df['Subject'].value_counts().head(5).index
        df_subset = self.df[self.df['Subject'].isin(top_subjects)].copy()
        
        # Create TF-IDF vectors
        print("  - Creating TF-IDF vectors...")
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_subset['final_corpus'].fillna(''))
        
        # Apply PCA
        print("  - Applying PCA...")
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(tfidf_matrix.toarray())
        
        # Add PCA results to dataframe
        df_subset['pca_1'] = pca_result[:, 0]
        df_subset['pca_2'] = pca_result[:, 1]
        df_subset['pca_3'] = pca_result[:, 2]
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df_subset,
            x='pca_1',
            y='pca_2',
            z='pca_3',
            color='Subject',
            hover_name='title',
            opacity=0.7,
            title=f"PCA Visualization of Top 5 Subjects ({len(df_subset)} books)",
            labels={'pca_1': 'PCA Component 1', 'pca_2': 'PCA Component 2', 'pca_3': 'PCA Component 3'}
        )
        
        fig.update_layout(
            height=800,
            width=1000
        )
        
        self.save_plotly(fig, "pca_visualization")
        
        # Create 2D scatter plot (PCA 1 vs PCA 2)
        fig2 = px.scatter(
            df_subset,
            x='pca_1',
            y='pca_2',
            color='Subject',
            hover_name='title',
            opacity=0.7,
            title=f"PCA Visualization (2D) of Top 5 Subjects ({len(df_subset)} books)",
            labels={'pca_1': 'PCA Component 1', 'pca_2': 'PCA Component 2'}
        )
        
        fig2.update_layout(
            height=600,
            width=1000
        )
        
        self.save_plotly(fig2, "pca_visualization_2d")
        
        return fig, fig2
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("🎨 Creating all visualizations...")
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Create visualizations
        self.create_subject_distribution()
        self.create_year_distribution()
        self.create_state_distribution()
        self.create_quality_score_distribution()
        self.create_subject_year_heatmap()
        self.create_wordcloud(column='final_corpus', title='Word Cloud of Enhanced Book Corpus')
        self.create_wordcloud(column='subject_keywords', title='Word Cloud of Subject Keywords')
        self.create_subject_wordclouds()
        self.create_quality_by_subject_boxplot()
        self.create_corpus_length_analysis()
        self.create_year_level_analysis()
        self.create_pca_visualization()
        
        print("✅ All visualizations created and saved to the visualization directory!")
        print(f"📁 Visualization directory: {self.visualization_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("📊 BOOK RECOMMENDATION ENGINE - DATA VISUALIZATION")
    print("=" * 60)
    
    visualizer = DataVisualizer()
    visualizer.load_data()
    visualizer.create_all_visualizations()
    
    print("\n" + "=" * 60)
    print("🏁 Visualization script completed!")
    print("=" * 60) 