import pandas as pd
import numpy as np

data = pd.read_csv('data/goemotions_1.csv')

str = 'admiration	amusement	anger	annoyance	approval	caring	confusion	curiosity	desire	disappointment	disapproval	disgust	embarrassment	excitement	fear	gratitude	grief	joy	love	nervousness	optimism	pride	realization	relief	remorse	sadness	surprise	neutral'

emotions_list = str.split('\t')

indexes = ['text']
for i in emotions_list:
    indexes.append(i)

df = data[indexes]

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

df["tokens"] = df["text"].apply(tokenizer.tokenize)

X = df.drop(emotions_list, axis=1)
X = X.drop('text', axis=1)

Y = df.drop('text', axis=1)
Y = Y.drop('tokens', axis=1)

# делаю специальный столбик с метками от 1 до 28 для каждой эмоции, т.е. у меня столбик emotion будет иметь значение от 1 до 28, т.е. соответсвовать какой-то из эмоций

df['emotion_label'] = np.where(df[emotions_list[0]] == 1, 1,
                               np.where(df[emotions_list[1]] == 1, 2,
                                        np.where(df[emotions_list[2]] == 1, 3,
                                                 np.where(df[emotions_list[3]] == 1, 4,
                                                          np.where(df[emotions_list[4]] == 1, 5,
                                                                   np.where(df[emotions_list[5]] == 1, 6,
                                                                            np.where(df[emotions_list[6]] == 1, 7,
                                                                                     np.where(df[emotions_list[7]] == 1,
                                                                                              8,
                                                                                              np.where(df[emotions_list[
                                                                                                  8]] == 1, 9,
                                                                                                       np.where(df[
                                                                                                                    emotions_list[
                                                                                                                        9]] == 1,
                                                                                                                10,
                                                                                                                np.where(
                                                                                                                    df[
                                                                                                                        emotions_list[
                                                                                                                            10]] == 1,
                                                                                                                    11,
                                                                                                                    np.where(
                                                                                                                        df[
                                                                                                                            emotions_list[
                                                                                                                                11]] == 1,
                                                                                                                        12,
                                                                                                                        np.where(
                                                                                                                            df[
                                                                                                                                emotions_list[
                                                                                                                                    12]] == 1,
                                                                                                                            13,
                                                                                                                            np.where(
                                                                                                                                df[
                                                                                                                                    emotions_list[
                                                                                                                                        13]] == 1,
                                                                                                                                14,
                                                                                                                                np.where(
                                                                                                                                    df[
                                                                                                                                        emotions_list[
                                                                                                                                            14]] == 1,
                                                                                                                                    15,
                                                                                                                                    np.where(
                                                                                                                                        df[
                                                                                                                                            emotions_list[
                                                                                                                                                15]] == 1,
                                                                                                                                        16,
                                                                                                                                        np.where(
                                                                                                                                            df[
                                                                                                                                                emotions_list[
                                                                                                                                                    16]] == 1,
                                                                                                                                            17,
                                                                                                                                            np.where(
                                                                                                                                                df[
                                                                                                                                                    emotions_list[
                                                                                                                                                        17]] == 1,
                                                                                                                                                18,
                                                                                                                                                np.where(
                                                                                                                                                    df[
                                                                                                                                                        emotions_list[
                                                                                                                                                            18]] == 1,
                                                                                                                                                    19,
                                                                                                                                                    np.where(
                                                                                                                                                        df[
                                                                                                                                                            emotions_list[
                                                                                                                                                                19]] == 1,
                                                                                                                                                        20,
                                                                                                                                                        np.where(
                                                                                                                                                            df[
                                                                                                                                                                emotions_list[
                                                                                                                                                                    20]] == 1,
                                                                                                                                                            21,
                                                                                                                                                            np.where(
                                                                                                                                                                df[
                                                                                                                                                                    emotions_list[
                                                                                                                                                                        21]] == 1,
                                                                                                                                                                22,
                                                                                                                                                                np.where(
                                                                                                                                                                    df[
                                                                                                                                                                        emotions_list[
                                                                                                                                                                            22]] == 1,
                                                                                                                                                                    23,
                                                                                                                                                                    np.where(
                                                                                                                                                                        df[
                                                                                                                                                                            emotions_list[
                                                                                                                                                                                23]] == 1,
                                                                                                                                                                        24,
                                                                                                                                                                        np.where(
                                                                                                                                                                            df[
                                                                                                                                                                                emotions_list[
                                                                                                                                                                                    24]] == 1,
                                                                                                                                                                            25,
                                                                                                                                                                            np.where(
                                                                                                                                                                                df[
                                                                                                                                                                                    emotions_list[
                                                                                                                                                                                        25]] == 1,
                                                                                                                                                                                26,
                                                                                                                                                                                np.where(
                                                                                                                                                                                    df[
                                                                                                                                                                                        emotions_list[
                                                                                                                                                                                            26]] == 1,
                                                                                                                                                                                    27,
                                                                                                                                                                                    np.where(
                                                                                                                                                                                        df[
                                                                                                                                                                                            emotions_list[
                                                                                                                                                                                                27]] == 1,
                                                                                                                                                                                        28,
                                                                                                                                                                                        0))))))))))))))))))))))))))))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


list_corpus = df["text"].tolist()
list_labels = df["emotion_label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                    random_state=40)

X_train_counts, count_vectorizer = cv(X_train)

"""#  ВНИМАНИЕ: вот этот блок лучше не перезапускать, т.к. обучение модельки будет идти порядка 10 минут"""

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', n_jobs=-1, random_state=40)

dependent = []
for i in emotions_list:
    dependent.append(i)
dependent.append('emotion_label')
dependent.append('tokens')
x = df[df.columns.drop(dependent)]

clf.fit(X_train_counts, y_train)

X_test_counts = count_vectorizer.transform(['This makes me glad, thank you for that'])

y_predicted_counts = clf.predict(X_test_counts)

"""some tests strings:"""

print(y_predicted_counts)

print(emotions_list[y_predicted_counts[0] - 1])

import collections

X_test_counts = count_vectorizer.transform(
    ['I appreciate it, that is good to know. I hope I will have to apply that knowledge one day'])

y_predicted_counts = clf.predict_proba(X_test_counts)

# print(y_predicted_counts)
emotion_and_prob = {}

for i in range(len(emotions_list)):
    emotion_and_prob[y_predicted_counts[0][i]] = emotions_list[i]

od = collections.OrderedDict(sorted(emotion_and_prob.items(), reverse=True))
print(od)

import joblib

joblib.dump(clf, 'model.pkl')
joblib.dump(count_vectorizer, 'vectorizer.pkl')
# это мы выводим эмоции в порядке уменьшения их вероятности в тексте

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
