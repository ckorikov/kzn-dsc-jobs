import warnings

from sklearn.svm import LinearSVC
from tpot import TPOTRegressor, TPOTClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
from matplotlib import pylab
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, log_loss, accuracy_score, average_precision_score, roc_curve, auc
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from preprocess.data_load import load_dataframe
from preprocess.text_preprocess import extract_words, pre_process_df, \
    vectorize_tr, with_topics, add_lda_train, extract_compensation
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from copy import copy
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def split_test_train(df):
    df_test = df[df.index > 600]
    df_train = df[df.index <= 600]
    return df_train, df_test


def get_target_train(df_train):
    binarizer = MultiLabelBinarizer()
    y_train = df_train.topic_list
    y_train = binarizer.fit_transform(y_train)
    return y_train, binarizer


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
    for i in range(0, y_test.shape[1]):
        if y_test.shape[1] == 1:
            acc = accuracy_score(y_test, predict)
        else:
            acc = accuracy_score(y_test[:, i], predict[:, i])
        if verbose:
            print(str(i) + ", accur: " + str(acc))
    if verbose:
        print("Hamming loss: " + str(h_loss))
        print("Log loss: " + str(l_loss))
    return h_loss

def tpot_train(cat, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, test_size=0.2)

    tpot = TPOTClassifier(generations=15, population_size=20, verbosity=5, n_jobs = -1, scoring='roc_auc')
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export(cat + '-pipeline.py')

def score_recall(category, est, X, y):
    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,
                                                        random_state=42)
    est.fit(X_train, y_train)
    y_score = est.decision_function(X_test)
    precision_recall_score(category, y_test, y_score)
    roc_score(category, y_test, y_score)


def roc_score(category, y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(category + ': Receiver operating characteristic example')
    plt.legend(loc="lower right")
    pylab.savefig(category + '-roc.png')
    plt.close()


def precision_recall_score(category, y_test, y_score):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(category + ':2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    pylab.savefig(category + '-pr.png')
    plt.close()
    print("Average precision: ", average_precision)


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


import random


class ReactionPredictor:
    categories = {
        # 'fork': ['fork', 'fork_and_knife'],
         'eww': ['eww', 'facepalm', 'noexcel', 'wat', 'hankey', 'nor', 'are_you_fucking_kidding_me'],
         'galera': ['galera', 'chains', 'rowing_galera'],
         'corporate': ['sberbank', 'putin', 'tinkoff', 'venheads', 'gref', 'putout', 'yandex'],
         'moneybag': ['moneybag', 'moneys', 'money_mouth_face'],
         'fire': ['fire', '+1', 'fireball', 'notbad', '+1::skin-tone-2', 'joy', '+1::skin-tone-6', 'good-enough',
                 'heavy_plus_sign'],
    }

    tpot_models = {
        'fire': make_pipeline(
                StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.9000000000000001, min_samples_leaf=6, min_samples_split=7, n_estimators=100)),
                RobustScaler(),
                StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.01, max_depth=6, max_features=0.05, min_samples_leaf=10, min_samples_split=4, n_estimators=100, subsample=0.6000000000000001)),
                StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=6, min_samples_split=7, n_estimators=100)),
                GaussianNB()
            ),
        'eww' : make_pipeline(
                make_union(
                    FunctionTransformer(copy),
                    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=1, max_features=0.7500000000000001, min_samples_leaf=7, min_samples_split=5, n_estimators=100, subsample=0.05))
                ),
                XGBClassifier(learning_rate=0.1, max_depth=8, min_child_weight=6, n_estimators=100, nthread=1, subsample=1.0)
            ),
        'galera': ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.8500000000000001, min_samples_leaf=19, min_samples_split=17, n_estimators=100),
        'corporate': make_pipeline(
                StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=10, min_samples_split=11, n_estimators=100)),
                StackingEstimator(estimator=LinearSVC(C=0.0001, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)),
                LinearSVC(C=0.01, dual=False, loss="squared_hinge", penalty="l1", tol=1e-05)
            ),
        'moneybag': make_pipeline(
                StackingEstimator(estimator=LogisticRegression(C=25.0, dual=True, penalty="l2")),
                StackingEstimator(estimator=LogisticRegression(C=1.0, dual=True, penalty="l2")),
                ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.45, min_samples_leaf=5, min_samples_split=12, n_estimators=100)
            )
    }

    cat_reply = {
        #'fork': ['fork.png', 'ban.png'],
        'fire': ['fire.png', 'not-bad.png', 'thumbs-up.png'],
        'eww': ['facepalm.png', 'noup.png'],
        'galera': ['galera.png'],
        'corporate': ['sberbank.png', 'venheads.png', 'putin.png'],
         'moneybag': ['money.png', 'moneybag.png'],
    }

    thinking_cats = ['thinking.png', 'thinking-parrot.png']

    words = ['вилк', 'зарплат', 'fork', 'money', 'деньг', 'sber', 'сбер', 'tinkoff', 'тиньк', 'X5',
             'retail', 'group', 'mail', 'больш', 'данн', 'big', 'data', 'middle', 'миддл', 'джун', 'juniо',
             'middle', 'senior', 'стажир', 'студ', 'выпускн', 'курс', 'молод', 'сеньор', 'архитект', 'фултайм',
             'младш', 'сотрудн', 'ШАД', 'SQL', 'excel', 'Java', 'JS', 'UI', 'C', 'C++', 'Hadoop', 'Spark', 'Front-end',
             'Backend', 'Frontend', 'молод', 'стартап', 'развива', 'коллектив', 'лидир', 'лидер', 'рекламн', 'агенств',
             'маркет', 'бизн', 'менедж', 'партнер', 'клиент', 'поставщик', 'услуг', 'департам', 'консалт', 'ритейл',
             'продаж', 'ранжир', 'Москва', 'моск', 'питер', 'Санкт', 'международ', 'Сколк', 'skol', 'академ', 'PhD',
             'гипотез', 'наук', 'science', 'Гос', 'грант', 'правит', 'ООО', 'подряд', 'бюрокр', 'график', 'удален',
             'ТК', 'парков', 'обед', 'кандид', 'требов', 'обязан', 'английск', 'офис', 'трудоустр', 'Соц', 'пакет',
             'карьерн', 'рост', 'аналит', 'продукт', 'сегмент', 'скорин', 'прогноз', 'отток', 'кредит', 'банк', 'A/B',
             'пространств', 'врем', 'ряд', 'фрод', 'tensorflow', 'theano', 'caffe', 'нейросет', 'keros', 'xgboost', 'R',
             'python', 'numpy', 'pandas', 'matplot', 'scikit', 'Tableau']

    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.lda_models = {}
        self.top_words = {}

    def fit(self):
        df = load_dataframe(self.categories)
        comp = extract_compensation(df)
        train_size = comp.shape[0]
        pp_tr_min = np.array([p[0] for p in comp]).reshape(train_size, 1)
        pp_tr_max = np.array([p[1] for p in comp]).reshape(train_size, 1)
        pp_train = pre_process_df(df)
        X_train, vectorizer = vectorize_tr(pp_train)
        train_w = extract_words(df, self.words)
        X_train = np.hstack((train_w, pp_tr_min, pp_tr_max, X_train.toarray()))
        for cat_key, cat_vals in self.categories.items():
            print("Topic: ", cat_key)
            #y_train, binarizer = get_target_train(df_cat)
            y_train = df[cat_key].values
            # X_train, lda_model = add_lda_train(X_train, vectorizer)
            clf = self.tpot_models[cat_key]
            #scores = cross_val_score(clf, X_train, y_train, scoring='r2', verbose=1, cv=5)
            #print("Scores :", scores)
            #
            clf.fit(X_train, y_train)
            score_recall(cat_key, LogisticRegression(solver='lbfgs', C=0.1), X_train, y_train)
            #tpot_train(cat_key, X_train, y_train)
            # num_top_features = 30
            # cols = self.words + ['min salary'] + ['max salary'] + vectorizer.get_feature_names() + ["lda"] * 300
            # top_features = pd.DataFrame.from_records(
            #    columns=['topic', 'top_words'],
            #    data=[(
            #        cls,
            #        [cols[i] for i in np.argpartition(coefs, -num_top_features)[-num_top_features:]]
            #    ) for cls, coefs in zip(cat_key, clf.coef_[:len(clf.classes_)])]
            # )
            # for row in top_features.values:
            #    self.top_words[cat_key] = row[1]
            #    print("Values: ", row[1])
            #    print("-----------------------------------------------")
            self.models[cat_key] = clf
            self.vectorizers[cat_key] = vectorizer

            # self.lda_models[cat_key] = lda_model

    def predict(self, text, threshold=0.4):
        res_react = np.array([])
        for cat_key, cat_vals in self.categories.items():
            model = self.models[cat_key]
            vectorizer = self.vectorizers[cat_key]
            # lda_model = self.lda_models[cat_key]

            df = pd.DataFrame([text], columns=['text'])
            words = extract_words(df, self.words)
            pp_text = pre_process_df(df)
            comp = extract_compensation(df)
            t_size = comp.shape[0]
            pp_min = np.array([p[0] for p in comp]).reshape(t_size, 1)
            pp_max = np.array([p[1] for p in comp]).reshape(t_size, 1)
            X_test = vectorizer.transform(pp_text, copy=True)
            # X_test = with_topics(X_test, lda_model)
            X_test = np.hstack((words, pp_min, pp_max, X_test.toarray()))

            pred_pr = model.predict(X_test)
            if pred_pr[0] > threshold:
                react = random.choice(self.cat_reply[cat_key])
                res_react = np.append(res_react, react)

        if len(res_react) == 0:
            return [random.choice(self.thinking_cats)]

        return res_react


if __name__ == '__main__':
    rp = ReactionPredictor()
    rp.fit()
    pickle.dump(rp, open("reaction_predictor.p", "wb"))
    str = "Мопед мой. Т.к. команда расширяется и мне добавляется менеджерских скиллов, ко мне в команду в Physician Partners https://www.linkedin.com/company/physician-partners, где я сейчас удаленно работаю, требуется *Middle / Senior Data scientist*  *О компании:* Компания занимается медстраховками в округе Нью-Йорк. Один наш экс-соотечественник ведёт для них проекты в области data science, то есть продаёт идеи по улучшению работы врачей и жизни пациентов. Наша задача помогать в этом. Я на проекте уже полгода, всё ОК *Место:* Полностью удаленная вакансия *Обязанности* Проект по медицинским данным. Нужно будет по истории заболеваний, анализов, назначений лекарств и прочему строить модели, которые будут помогать диагностировать новые болезни, прогнозировать попадание человека в госпиталь и т.д. Так же придется заниматься аналитикой, то есть строить графики, проверять гипотезы, делать небольшие презентации. Соотношение задач модели / аналитика примерно 70 / 30. В начале аналитики может быть побольше, т.к. надо убеждать бизнес в том, что проблема есть, что модель работает, приводить реальные примеры.  *Требования* · опыт работы не менее 2 лет в области Data Science, хорошее знакомство с классикой в виде стат. анализа деревьев, бустинга, кластеризации, начальные/средние знания по DL, т.к. в планах пробовать что-то поинтереснее xgboost, но без фанатизма; · личные проекты и соревнования как плюс к карме; · продвинутые знания Python, основных библиотек для работы с данными; · умение работать в Linux/Unix Shell; · отсутствие страха перед Git, Wiki, Jira и другими средствами командной работы; · Docker (модели заливаем на google cloud), нужно уметь презентовать модель в виде простого web сервиса типа Flask, чтобы доктора могли посмотреть; · опыт написания SQL запросов, нужно будет копаться в данных; · способность презентовать результаты своей работы."
    print(rp.predict(str))
