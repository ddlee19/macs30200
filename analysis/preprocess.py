import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy
import spacy.symbols

# cluster detection
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

import matplotlib.pyplot as plt #For graphics
import matplotlib.cm #Still for graphics
import seaborn as sns #Makes the graphics look nicer

# load spacy model
nlp = spacy.load('en_core_web_sm')

def preprocess(input=''):
    '''
    Preprocess csv file of Reddit data

    -input: name of input csv file
    -output: name of output csv file
    '''

    comments_df = pd.read_csv(input, header=None, names=['username','flair_text','body'])

    # drop rows with deleted text body
    deleted_rows = comments_df[comments_df['body'].isin(['[deleted]','[removed]'])]
    comments_df.drop(deleted_rows.index, inplace=True)
    
    # drop rows that have been filtered by AutoModerator
    moderated_rows = comments_df[comments_df['username']=='AutoModerator']
    comments_df.drop(moderated_rows.index, inplace=True)
    comments_df.reset_index(drop=True)

    # tokenize
    comments_df['tokens'] = comments_df['body'].apply(lambda x: tokenize_str(x))

    # normalize
    comments_df['normalized_tokens'] = comments_df['tokens'].apply(lambda x: normalize_tokens(x))

    # add count columns
    comments_df['word_count'] = comments_df['tokens'].apply(lambda x: len(x))
    comments_df['normalized_tokens_count'] = comments_df['normalized_tokens'].apply(lambda x: len(x))

    # output file path
    output_path = input[0:-4] + '_processed' + input[-4:]
    comments_df.to_csv(output_path)

    return f'output filepath: {output_path}'


def tokenize_str(str_, nlp=nlp):
    '''
    Tokenize strings using Spacy model (preserves Reddit usernames)

    -str_: string to tokenize
    -nlp: Spacy model
    '''

    nlp.tokenizer.token_match = re.compile(r'u/{1}\w{3,23}').match # match reddit usernames

    tokenized = []
    doc = nlp(str_)
    for token in doc:
        if not token.is_punct and len(token.text.strip()) > 0:
            tokenized.append(token.text)
    return tokenized


def normalize_tokens(word_list, nlp=nlp, extra_stop_words=[]):
    '''
    1. Change tokens to lowercase
    2. Drop non-word tokens
    3. Remove stop-words
    4. Lemmatize tokens

    -word_list: list of words/tokens
    -nlp: Spacy model
    '''

    '''
    BELOW CODE: complete writing and uncomment if need to add extra stop words
    # create list of stop-words to remove using word counts
    count_dict = {}
    for word in df[text_col].sum():
        word = word.lower()
        if word in count_dict:
            count_dict[word]+=1
        else:
            count_dict[word]=1

    # sort tuples of word and frequency pairs by frequency
    word_counts = sorted(count_dict.items(), key = lambda x: x[1], reverse=True)
    # create list of stop words that are more frequent than the 
    stop_words_freq = 
    '''
    # list of normalized words to return
    normalized = []

    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list])

    doc = nlp(word_list.lower())

    # add lexeme property of stopwords
    if len(extra_stop_words) > 0:
        for stopword in extra_stop_words:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True
        
    for w in doc:
        # if not stop word or punctuation, add it to normalized
        if not w.text.isspace() and not w.is_stop and not w.is_punct and not w.like_num:
            normalized.append(str(w.lemma_))

    return normalized