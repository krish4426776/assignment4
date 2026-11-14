
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download English stopwords
nltk.download("stopwords")


def preprocess_sentence(sentence):
    # Split the sentence into words and punctuation
    tokens = wordpunct_tokenize(sentence)

    # Remove common English words and punctuation
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        word for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]

    # Reduce words to their root forms
    stemmer = PorterStemmer()

    stemmed_words = [
        stemmer.stem(word.lower())
        for word in filtered_tokens
    ]

    # Display the results
    print("Original Tokens:", tokens)
    print("Tokens Without Stopwords:", filtered_tokens)
    print("Stemmed Words:", stemmed_words)


# Sentence provided in the assignment
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

preprocess_sentence(sentence)
