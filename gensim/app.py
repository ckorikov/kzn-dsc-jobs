import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from preprocess.data_load import load_dataframe
from preprocess.text_preprocess import extract_words, pre_process_df, \
    vectorize_tr, with_topics, add_lda_train

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
        'fork': ['fork', 'fork_and_knife'],
        'fire': ['fire', '+1', 'fireball', 'notbad', '+1::skin-tone-2', 'joy', '+1::skin-tone-6', 'good-enough',
                 'heavy_plus_sign'],
        'eww': ['eww', 'facepalm', 'noexcel', 'wat', 'hankey', 'nor', 'are_you_fucking_kidding_me'],
        'galera': ['galera', 'chains', 'rowing_galera'],
        'corporate': ['sberbank', 'putin', 'tinkoff', 'venheads', 'gref', 'putout', 'yandex'],
        'money': ['moneybag', 'moneys', 'money_mouth_face']
    }

    thinking_cats = [':thinking_face:', ':thinking_alot:', ':confusedparrot:']

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
        for cat_key, cat_vals in self.categories.items():
            df = load_dataframe(cat_key, cat_vals)
            train_w = extract_words(df, self.words)
            pp_train = pre_process_df(df)
            pp_tr_t = np.array([p[0] for p in pp_train])
            train_size = pp_tr_t.shape[0]
            pp_tr_min = np.array([p[1] for p in pp_train]).reshape(train_size, 1)
            pp_tr_max = np.array([p[2] for p in pp_train]).reshape(train_size, 1)

            X_train, vectorizer = vectorize_tr(pp_tr_t)
            y_train, binarizer = get_target_train(df)
            # X_train, lda_model = add_lda_train(X_train, vectorizer)
            X_train = np.hstack((train_w, pp_tr_min, pp_tr_max, X_train.toarray()))
            clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.1)
            clf.fit(X_train, y_train)

            num_top_features = 30
            cols = self.words + ['min salary'] + ['max salary'] + vectorizer.get_feature_names() + ["lda"] * 300
            top_features = pd.DataFrame.from_records(
                columns=['topic', 'top_words'],
                data=[(
                    cls,
                    [cols[i] for i in np.argpartition(coefs, -num_top_features)[-num_top_features:]]
                ) for cls, coefs in zip(cat_key, clf.coef_[:len(clf.classes_)])]
            )
            for row in top_features.values:
                self.top_words[cat_key] = row[1]
                print("Topic: ", cat_key)
                print("Values: ", row[1])
                print("-----------------------------------------------")

            self.models[cat_key] = clf
            self.vectorizers[cat_key] = vectorizer

            # self.lda_models[cat_key] = lda_model

    def predict(self, text, threshold=0.66):
        res_react = np.array([])
        for cat_key, cat_vals in self.categories.items():
            model = self.models[cat_key]
            vectorizer = self.vectorizers[cat_key]
            # lda_model = self.lda_models[cat_key]

            df = pd.DataFrame([text], columns=['text'])
            words = extract_words(df, self.words)
            pp = pre_process_df(df)
            pp_text = np.array([p[0] for p in pp])
            siz = pp_text.shape[0]
            pp_min = np.array([p[1] for p in pp]).reshape(siz, 1)
            pp_max = np.array([p[2] for p in pp]).reshape(siz, 1)
            X_test = vectorizer.transform(pp_text, copy=True)
            # X_test = with_topics(X_test, lda_model)
            X_test = np.hstack((words, pp_min, pp_max, X_test.toarray()))

            pred_pr = model.predict_proba(X_test)
            if pred_pr[0][0] > threshold:
                react = ':' + random.choice(cat_vals) + ':'
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
