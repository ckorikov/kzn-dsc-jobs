import gensim
from nltk.corpus import stopwords
from nltk.stem.lancaster import *
from nltk.stem.snowball import RussianStemmer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = stopwords.words('russian')


def topic_list_from_binarized(binarized, topics):
    return [topic for (has, topic) in zip(binarized, topics) if has == 1]


def single_label(df):
    return df.apply(lambda row: row['topic_list'][0], axis=1).values


def binarized(df):
    return sparse.csr_matrix(df.apply(lambda row: row['topics_binarized'], axis=1).values.tolist())


def lower_case(string):
    return string.lower()


def fix_lt(string):
    # fix the HTML-escaped less-than sign
    return re.sub(r'&lt;', '<', string)


def replace_non_alphanumeric_with_space(string):
    # replace punctuation and different whitespace with space character
    return re.sub(r'[^\w0-9\s]', ' ', string)


def strip_punctuation(string):
    # remove punctuation
    return re.sub(r'[^\w0-9\s]', ' ', string)


def remove_stop_words(string, stop_words):
    return ' '.join([word for word in re.split(' ', string) if not word in stop_words])


def replace_numeric_with_literal(string, literal='<num> '):
    return re.sub(r'([0-9]+ ?)+', literal, string)


def compact_whitespace(string):
    return re.sub(r'\s+', ' ', string)


def stem(string, stemmer):
    return ' '.join([stemmer.stem(word) for word in re.split(' ', string) if not word in stop_words])


def lemmatize(string, lemmatizer):
    return ' '.join([lemmatizer.lemmatize(word) for word in re.split(' ', string) if not word in stop_words])


def pre_process(string):
    s = string
    s = lower_case(string)
    s = fix_lt(s)
    s = strip_punctuation(s)
    #s = remove_stop_words(s, stop_words)
    #s = compact_whitespace(s)
    #s = replace_numeric_with_literal(s)
    stemmer = RussianStemmer()
    s = stem(s, stemmer)
    return s.strip()


def pre_process_df(df):
    return df.apply(lambda row: pre_process(row['text']), axis=1).values


def pre_process_both(df_train, df_test):
    pp_train = pre_process_df(df_train)
    pp_test = pre_process_df(df_test)
    return pp_train, pp_test


def vectorize(pp_train, pp_test):
    vectorizer = TfidfVectorizer(
        sublinear_tf=False,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{2,}',
        ngram_range=(1, 2),
        max_features=1510,
        max_df=0.7,
        norm=None
    )

    vectorizer.fit(pp_train)

    X_train = vectorizer.transform(pp_train, copy=True)
    X_test = vectorizer.transform(pp_test, copy=True)
    return X_train, X_test, vectorizer


def sklearn2gensim(X):
    return gensim.matutils.Sparse2Corpus(X.T)


def gensim2sklearn(ldayed):
    return gensim.matutils.corpus2csc([
        topics
        for topics, word_topics, word_topics_weights
        in ldayed
    ]).T


def with_topics(X, lda):
    X_gensim = sklearn2gensim(X)
    ldayed = lda[X_gensim]
    X_topics = gensim2sklearn(ldayed)
    return sparse.hstack((X, X_topics)).A


def add_lda(X_train, X_test, vectorizer):
    id2word = {v: k for k, v in vectorizer.vocabulary_.items()}
    X_train_gensim = sklearn2gensim(X_train)
    lda = gensim.models.ldamodel.LdaModel(
        corpus=X_train_gensim,
        id2word=id2word,
        num_topics=57,
        per_word_topics=True
    )

    X_train_with_topics = with_topics(X_train, lda)
    X_test_with_topics = with_topics(X_test, lda)
    return X_train_with_topics, X_test_with_topics
