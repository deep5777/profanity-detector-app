import streamlit as st
from nltk.corpus import stopwords
import gensim
import pandas as pd
from gensim.models import Word2Vec
import re
import pickle
pd.options.mode.copy_on_write = True

df = pd.read_csv("profanity_en.csv")
duplicate_data = df[df.duplicated()]
df = df[["text", "canonical_form_1", "category_1", "severity_rating", "severity_description"]]

# Remove stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('a')
stop_words.add('of')


def remove_stopwords(sentence):
    tokens = sentence.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ''.join(filtered_tokens)


df['text_without_stopwords'] = df['text'].apply(remove_stopwords)
shuffled_df = df.sample(frac=1)
shuffled_df.to_csv("shuffled.csv", index=False)

review_text = shuffled_df.text.apply(gensim.utils.simple_preprocess)


def custom_preprocess(text, retain_chars=None):
    """
    Custom preprocess function to retain specific characters and tokenize the text.

    :param text: The input text to preprocess.
    :param retain_chars: List of characters to retain in the text.
    :return: List of tokens.
    """
    # Create a regex pattern to retain the specific characters
    if retain_chars is None:
        retain_chars = ['@', '+', '$', '%', '*']
    retain_pattern = ''.join(map(re.escape, retain_chars))

    # Define a regex pattern to match words and specified characters
    pattern = re.compile(rf'[a-zA-Z{retain_pattern}]+')

    # Find all tokens that match the pattern
    tokens = pattern.findall(text.lower())

    # Remove tokens that contain numbers
    tokens = [token for token in tokens if not any(char.isdigit() for char in token)]

    # Remove single-character tokens
    tokens = [token for token in tokens if len(token) > 1]

    return tokens


# Apply the custom preprocessing function to the DataFrame column
df['processed_column'] = df['text_without_stopwords'].apply(custom_preprocess)

model = gensim.models.Word2Vec(
    window=1,
    min_count=1,
    workers=4
)
model.build_vocab(df['processed_column'], progress_per=500)
model.train(df['processed_column'], total_examples=model.corpus_count, epochs=model.epochs)

# Assuming `model` is your trained Word2Vec model
with open("word2vec_profanity.pkl", "wb") as file:
    pickle.dump(model, file)

# Load the Word2Vec model
with open("word2vec_profanity.pkl", "rb") as file:
    model = pickle.load(file)


def check_profanity(user_input, model, similarity_threshold=0.3):
    user_input = user_input.lower()
    user_input = user_input.replace('_', ' ').replace(',', ' ')
    user_input = user_input.split()
    profane_words = []

    for word in user_input:
        try:
            similarity = model.wv.most_similar(word)
            if similarity[0][1] >= similarity_threshold:
                profane_words.append(word)
        except KeyError:
            # Word not in vocabulary, continue checking next word
            continue

    if profane_words:
        st.write("Profane words : " + ' , '.join(profane_words))
        return 'Profanity Found'
    else:
        return 'Profanity not Found'

st.title('Profanity Detector')
# Take input from the user
user_input = st.text_area("Enter some text:",height=100)
user_input1 = user_input

if st.button('Predict'):
    # Print the result
    st.write("Your Input : " + user_input1)
    st.header("Result : " + check_profanity(user_input, model))
