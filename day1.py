from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Display the TF-IDF matrix
print(X.toarray())

# Display the feature names
print(vectorizer.get_feature_names_out())