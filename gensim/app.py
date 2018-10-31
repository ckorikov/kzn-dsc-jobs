import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from preprocess.data_load import load_dataframe
from preprocess.text_preprocess import vectorize, add_lda, pre_process_both


def split_test_train(df):
    df_test = df[df.index > 600]
    df_train = df[df.index <= 600]
    return df_train, df_test


def get_target(df_train, df_test):
    binarizer = MultiLabelBinarizer()
    y_train = df_train.topic_list
    y_test = df_test.topic_list
    y_train = binarizer.fit_transform(y_train)
    y_test = binarizer.transform(y_test)
    return y_test, y_train, binarizer


def score(y_test, predict, predict_proba, verbose=True):
    h_loss = hamming_loss(y_test, predict)
    l_loss = log_loss(y_test, predict_proba, normalize=True)
    if verbose:
        print("Hamming loss: " + str(h_loss))
        print("Log loss: " + str(l_loss))
    return h_loss


def fit_predict(X_train, X_test, y_train, y_test):
    X_f = np.vstack((X_train, X_test))
    y_f = np.vstack((y_train, y_test))
    classif = OneVsRestClassifier(RandomForestClassifier())
    params = {
        "estimator__min_samples_split": [2, 3, 5, 8, 13],
        "estimator__min_samples_leaf": [1, 2, 3, 5, 8, 13],
        "estimator__max_depth": [None, 3, 5, 8, 13],
        "estimator__max_features": [None, 'auto', 'log2']
    }
    classif = GridSearchCV(classif, scoring='neg_log_loss', param_grid=params, verbose=10)
    classif.fit(X_f, y_f)
    return classif


def fit_rfc(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(max_depth=3, max_features=None, min_samples_leaf=1, min_samples_split=5)
    est = OneVsRestClassifier(rfc)
    est.fit(X_train, y_train)
    pred = est.predict(X_test)
    pred_proba = est.predict_proba(X_test)
    score(y_test, pred, pred_proba)
    return est


def fit_log_reg(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.1)
    est = OneVsRestClassifier(lr)
    est.fit(X_train, y_train)
    pred = est.predict(X_test)
    pred_proba = est.predict_proba(X_test)
    score(y_test, pred, pred_proba)
    return est


# df, topics = load_dataframe(5)
# df_train, df_test = split_test_train(df)
# pp_train, pp_test = pre_process_both(df_train, df_test)
# X_train, X_test, vectorizer = vectorize(pp_train, pp_test)
# X_t_train, X_t_test = add_lda(X_train, X_test, vectorizer)
# y_test, y_train, binarizer = get_target(df_train, df_test)


def train_evaluate(C):
    if X_t_train.shape[1] != X_t_test.shape[1]:
        return -100
    est = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=C)
    est = OneVsRestClassifier(est)
    est.fit(X_t_train, y_train)
    pred = est.predict(X_t_test)
    pred_proba = est.predict_proba(X_t_test)
    return score(y_test, pred, pred_proba, verbose=True)


def get_params():
    return {
        "C": (0.1, 1)
    }


def b_opt():
    num_rounds = 3000
    random_state = 2016
    num_iter = 25
    init_points = 5
    params = {
        'eta': 0.1,
        'silent': 1,
        'eval_metric': 'mae',
        'verbose_eval': True,
        'seed': random_state
    }

    xgbBO = BayesianOptimization(train_evaluate, get_params())

    xgbBO.maximize(init_points=init_points, n_iter=num_iter)
    return xgbBO


if __name__ == '__main__':
    # bo = b_opt()

    df, topics = load_dataframe(10)
    df_train, df_test = split_test_train(df)
    pp_train, pp_test = pre_process_both(df_train, df_test)
    X_train, X_test, vectorizer = vectorize(pp_train, pp_test)
    X_t_train, X_t_test = add_lda(X_train, X_test, vectorizer)
    y_test, y_train, binarizer = get_target(df_train, df_test)
    clf1 = fit_rfc(X_t_train, X_t_test, y_train, y_test)
    clf = fit_log_reg(X_t_train, X_t_test, y_train, y_test)
    top_features = []

    num_top_features = 10  # min(10, clf.coef_.shape[0])
    top_features = pd.DataFrame.from_records(
        columns=['topic', 'top_words'],
        data=[(
            cls,
            [vectorizer.get_feature_names()[i] for i in np.argpartition(coefs, -num_top_features)[-num_top_features:]]
        ) for cls, coefs in zip(topics, clf.coef_[:len(clf.classes_)])]
    )
    for row in top_features.values:
        print("Topic: ", row[0])
        print("Values: ", row[1])
        print("-----------------------------------------------")
