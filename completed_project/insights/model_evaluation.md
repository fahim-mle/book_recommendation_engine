# Book Recommendation Engine: Model Evaluation

## Evaluation Date: 2025-07-04 18:38:01

## Dataset Information
- **Size:** 1071 books
- **Features:** 5000 TF-IDF features
- **Corpus Used:** final_corpus

## Model Performance Metrics
- **Coverage:** 0.2941 - *Percentage of books that get recommended*
- **Diversity:** 0.4492 - *Average dissimilarity between recommendations (higher is better)*
- **Novelty:** 0.7748 - *How surprising/unexpected the recommendations are (higher is better)*
- **Average Similarity Score:** 0.5508 - *Average similarity between query and recommendations*

## Sample Recommendations

### Based on Book Title
**Query:** "Spelling mastery / Robert Dixon, Siegfried Engelmann, Mary Meier Bauer"

**Recommendations:**
1. "Spelling mastery / Robert Dixon, Siegfried Engelmann, Mary Meier Bauer (ISBN: 9780076044825)" (Similarity: 0.9621)
2. "Spelling mastery / Robert Dixon, Siegfried Engelmann, Mary Meier Bauer (ISBN: 9780076044849)" (Similarity: 0.9515)
3. "Spelling mastery / Robert Dixon, Siegfried Engelmann, Mary Meier Bauer (ISBN: 9780076044856)" (Similarity: 0.9515)

### Based on ISBN
Error generating ISBN-based recommendations: Book with ISBN '9781108562379' not found in the dataset.

### Based on Subject and Year Level
**Query:** LANGUAGES (Year 12)

**Recommendations:**
1. "Wakatta! / by David Jaffray and Masumi Sorrell (ISBN: 9781740200523)" 
2. "The student's Catullus / [edited by] Daniel H. Garrison" 
3. "Tapis volant /  Jane Zemiro, Alan Chamberlain (ISBN: 9780170129404)" 


## Interpretation of Results

The content-based recommendation engine demonstrates reasonable performance across the evaluation metrics:

1. **Coverage:** The model recommends approximately 29% of the available books, showing good catalog utilization.

2. **Diversity:** With a diversity score of 0.45, the recommendations show moderate variety, balancing between similar books and diverse options.

3. **Novelty:** The high novelty score of 0.77 indicates that the model recommends books that are relatively uncommon, avoiding popularity bias.

4. **Similarity:** The average similarity score of 0.55 suggests that recommendations are relevant to the query items while not being too obvious or identical.

## Limitations and Considerations

- The model relies entirely on text features and doesn't incorporate user preferences or behavioral data.
- Recommendations are based on textual similarity, which may miss deeper semantic relationships between books.
- The quality of recommendations depends heavily on the richness of the text corpus used for feature extraction.
- Limited data for some books may result in less accurate recommendations.
- The model doesn't account for pedagogical relationships or curriculum alignment beyond what's captured in the text features.
