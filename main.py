import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Sample product review data (you can replace this with your dataset)
reviews = [
    ("This product is amazing!", "positive"),
    ("I don't like this product.", "negative"),
    ("Great value for the price.", "positive"),
    ("The quality is poor.", "negative"),
    ("Highly recommend it!", "positive"),
    ("not so good", "negative")  # Include this example explicitly
]

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

# Preprocess reviews
preprocessed_reviews = [(preprocess_text(review), sentiment) for review, sentiment in reviews]

# Extract words and their corresponding sentiments from reviews
word_sentiments = {}
for review, sentiment in preprocessed_reviews:
    for word in review.split():
        if word not in word_sentiments:
            word_sentiments[word] = sentiment

# Vectorize text data
X = list(word_sentiments.keys())
y = list(word_sentiments.values())

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_vectorized, y)

# Function to classify input sentence based on individual words with negation handling
def classify_sentence(sentence):
    preprocessed_input = preprocess_text(sentence)
    words = preprocessed_input.split()
    positive_count = 0
    negative_count = 0
    negation_flag = False

    for word in words:
        if word in ["not", "no", "never", "none"]:
            negation_flag = True
            continue

        if word in vectorizer.vocabulary_:
            word_vectorized = vectorizer.transform([word])
            prediction = classifier.predict(word_vectorized)[0]

            if negation_flag:
                prediction = "negative" if prediction == "positive" else "positive"
                negation_flag = False

            if prediction == 'positive':
                positive_count += 1
            elif prediction == 'negative':
                negative_count += 1

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# User input
user_input = input("Enter your sentence: ")
prediction = classify_sentence(user_input)
print("Prediction:", prediction)
