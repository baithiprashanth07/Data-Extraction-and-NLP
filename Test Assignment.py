import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from textstat.textstat import textstat

# Load the input.xlsx file
input_df = pd.read_excel('input.xlsx')

# Create a directory to store the extracted article texts
if not os.path.exists('articles'):
    os.makedirs('articles')

# Extract article text for each URL
for index, row in input_df.iterrows():
    url = row['URL']
    url_id = row['URL_ID']
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = ''
    for p in soup.find_all('p'):
        article_text += p.text + '\n\n'
    with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as f:
        f.write(article_text.strip())

# Load the extracted article texts
article_texts = []
for file in os.listdir('articles'):
    with open(f'articles/{file}', 'r', encoding='utf-8') as f:
        article_texts.append(f.read())

# Initialize the output data
output_data = []

# Perform textual analysis for each article
for i, article_text in enumerate(article_texts):
    url_id = input_df.iloc[i]['URL_ID']
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(article_text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores['neu']
    
    # Tokenization
    words = word_tokenize(article_text)
    sentences = sent_tokenize(article_text)
    
    # Calculate AVG SENTENCE LENGTH
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    
    # Calculate PERCENTAGE OF COMPLEX WORDS
    complex_words = [word for word in words if textstat.flesch_reading_ease(word) < 60]
    percentage_of_complex_words = len(complex_words) / len(words) * 100
    
    # Calculate FOG INDEX
    fog_index = textstat.gunning_fog(article_text)
    
    # Calculate AVG NUMBER OF WORDS PER SENTENCE
    avg_words_per_sentence = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    
    # Calculate COMPLEX WORD COUNT
    complex_word_count = len(complex_words)
    
    # Calculate WORD COUNT
    word_count = len(words)
    
    # Calculate SYLLABLE PER WORD
    syllable_per_word = sum(textstat.syllable_count(word) for word in words) / len(words)
    
    # Calculate PERSONAL PRONOUNS
    personal_pronouns = [word for word in words if word in ['I', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs']]
    personal_pronouns_count = len(personal_pronouns)
    
    # Calculate AVG WORD LENGTH
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Append the output to the data
    output_data.append({
        'URL_ID': url_id, 
        'POSITIVE SCORE': positive_score, 
        'NEGATIVE SCORE': negative_score, 
        'POLARITY SCORE': polarity_score, 
        'SUBJECTIVITY SCORE': subjectivity_score, 
        'AVG SENTENCE LENGTH': avg_sentence_length, 
        'PERCENTAGE OF COMPLEX WORDS': percentage_of_complex_words, 
        'FOG INDEX': fog_index, 
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence, 
        'COMPLEX WORD COUNT': complex_word_count, 
        'WORD COUNT': word_count, 
        'SYLLABLE PER WORD': syllable_per_word, 
        'PERSONAL PRONOUNS': personal_pronouns_count, 
        'AVG WORD LENGTH': avg_word_length
    })

# Create the output DataFrame
output_df = pd.DataFrame(output_data)

# Save the output to a CSV file
output_df.to_csv('output.csv', index=False)
