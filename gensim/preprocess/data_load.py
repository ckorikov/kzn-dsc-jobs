import pandas as pd
import numpy as np
import json
import os

r_dict = {}

categories = {
    'fork': ['ban', 'fork', 'fork_and_knife'],
    'fire': ['+1', 'fire', 'fireball', 'notbad', '+1::skin-tone-2', 'joy', '+1::skin-tone-6', 'heavy_plus_sign'],
    'galera': ['galera', 'eww', 'facepalm', 'noexcel', 'chains', 'wat', 'hankey', 'nor', 'rowing_galera', 'good-enough',
               'are_you_fucking_kidding_me'],
    'corporate': ['sberbank', 'putin', 'tinkoff', 'venheads', 'gref', 'putout', 'yandex'],
    'money': ['moneybag', 'moneys', 'money_mouth_face']
}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        return data


def fill_r_dict(d):
    for key, value in d.items():
        if key not in r_dict:
            r_dict[key] = 0
        r_dict[key] = r_dict[key] + value


def extract_category(cats, d):
    for c in cats:
        if c not in d:
            continue
        if d[c] > 1:
            return 1
    return 0

def topic_list_from_binarized(binarized, all_topics):
    return [topic for (has, topic) in zip(binarized, all_topics) if has == 1]


def load_dataframe(indx):
    react_size = 50
    dataset = read_json('dataset.json')
    dataset = np.array(dataset)

    reactions = [d['reactions'] for d in dataset]
    [fill_r_dict(r) for r in reactions]
    r_list = list(r_dict.items())
    reacts = sorted(r_list, key=lambda tup: -tup[1])[0:react_size]
    all_topics = [r[0] for r in reacts]
    df = pd.DataFrame()
    df['text'] = [d['text'] for d in dataset]
    cats = list(categories.keys())[indx:indx+1]
    for cat_key in cats:
        df[cat_key] = [extract_category(categories[cat_key], r) for r in reactions]

    print('\n', 'num topics:', len(all_topics))
    pd.DataFrame(all_topics, columns=['topic name']).head(15)

    topics_binarized = np.array(df.values[:, 1:])
    topic_lists = [topic_list_from_binarized(binarized, all_topics) for binarized in topics_binarized]
    df = df[['text']]
    df = df.assign(topic_list=topic_lists)
    df = df.assign(topics_binarized=topics_binarized.tolist())
    df = df.assign(num_topics=np.array([len(lst) for lst in topic_lists]))
    return df, cats
