# Book Recommendation Engine: Model Evaluation

## Evaluation Date: 2025-07-07 20:05:01

## Dataset Information
- **Size:** 1071 books
- **Features:** 5000 TF-IDF features
- **Corpus Used:** final_corpus

## Model Performance Metrics
- **Coverage:** 0.3059 - *Percentage of books that get recommended*
- **Diversity:** 0.4928 - *Average dissimilarity between recommendations (higher is better)*
- **Novelty:** 0.7748 - *How surprising/unexpected the recommendations are (higher is better)*
- **Average Similarity Score:** 0.5072 - *Average similarity between query and recommendations*

## Sample Recommendations

### Based on Book Title
**Query:** "Biology in focus. skills and assessment workbook / Julie Fraser, Kirsten Prior, Evan Roberts"

**Recommendations:**
1. "Physics in focus. skills and assessment workbook / Adam Sloan, Edward Baker, Darren Goossens, Owen Hamerton" (Similarity: 0.7064)
2. "nan" (Similarity: 0.5151)
3. "nan" (Similarity: 0.5151)

### Based on ISBN
**Query ISBN:** 9780076044856

**Recommendations:**
1. "Spelling mastery / Robert Dixon, Siegfried Engelmann, Mary Meier Bauer (ISBN: 9780076044863)" (Similarity: 1.0000)
2. "Spelling mastery / Robert Dixon, Siegfried Engelmann, Mary Meier Bauer (ISBN: 9780076044856)" (Similarity: 1.0000)
3. "Spelling mastery / Robert Dixon, Siegfried Engelmann, Mary Meier Bauer (ISBN: 9780076044825)" (Similarity: 0.9899)

### Based on Subject and Year Level
**Query:** ENGLISH (Year 12)

**Recommendations:**
1. "Selected poems [Kenneth Slessor]" 
2. "Perfume : the story of a murderer / Patrick Süskind ; translated from the German by John E. Woods" 
3. "ATAR Notes Text Guide : Burial rites by Hannah Kent / Morgaine Sharp" 


## Interpretation of Results

The content-based recommendation engine demonstrates reasonable performance across the evaluation metrics:

1. **Coverage:** The model recommends approximately 31% of the available books, showing good catalog utilization.

2. **Diversity:** With a diversity score of 0.49, the recommendations show moderate variety, balancing between similar books and diverse options.

3. **Novelty:** The high novelty score of 0.77 indicates that the model recommends books that are relatively uncommon, avoiding popularity bias.

4. **Similarity:** The average similarity score of 0.51 suggests that recommendations are relevant to the query items while not being too obvious or identical.

## Limitations and Considerations

- The model relies entirely on text features and doesn't incorporate user preferences or behavioral data.
- Recommendations are based on textual similarity, which may miss deeper semantic relationships between books.
- The quality of recommendations depends heavily on the richness of the text corpus used for feature extraction.
- Limited data for some books may result in less accurate recommendations.
- The model doesn't account for pedagogical relationships or curriculum alignment beyond what's captured in the text features.
