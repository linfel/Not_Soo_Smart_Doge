from final_bot_config import BOT_CONFIG

# Готовим датасет
dataset = []  # [[example, intent], ...]

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        dataset.append([example, intent])

# print(dataset)

# разделим его на списки примеров и намерений
X_text = [x for x, y in dataset]
y = [y for x, y in dataset]
# print(X_text)
# print(y)

# Векторизация
# Конвертируем примеры в вектора из чисел.
#
# Можно выбрать любой векторайзер. Также можно произвести обработку текстов заранее (например привести все слова к
# начальной форме).
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer(lowercase=True, ngram_range=(3, 3), analyzer='char_wb')
X = vectorizer.fit_transform(X_text)  # вектора примеров

# print(vectorizer.get_feature_names())  # посмотреть на словарь (все слова из датасета)
# print(X.toarray())  # посмотреть на сами вектора

# Обучение модели
# Можно выбрать любой алгоритм и настроить его параметры.
# В зависимости от способа векторизации, выбранного алгоритма и параметров мы получим соответствующее качество работы
# исходной модели. Ее качество также зависит от датасета.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

tfift_transformer = TfidfTransformer()
x_tf = tfift_transformer.fit_transform(X)
X= x_tf

clf = LogisticRegression()
clf.fit(X, y)

clf = SVC(probability=True)
# SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
#     max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
#     verbose=False)
clf.fit(X, y)

# Предсказание намерений
v = vectorizer.transform(['Привет :)', 'как дела', 'что там с погодой?'])
# print(v.toarray())     # вектора, соответствующие репликам
# print(clf.predict(v))  # предсказания намерений, которые соответствуют репликам

# Также можем предсказать вероятность соответствия реплик каждому намерению

# print(clf.classes_)  # намерения
# print(clf.predict_proba(v))  # вероятности для намерений

# Оценка качества модели
# Делим выборку на обучающую и тестовую. На обучаещей обучаем, на тестовой - считаем количество попаданий
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(X_train.shape, X_test.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  # доля попаданий в правильный класс (от 0 до 1)

# Каждый раз выборка делится на 2 части по разному - это влияет на качество.
# Чтобы получить усредненный результат, сделаем подсчет несколько раз:

scores = []

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

print(sum(scores) / 100)

























# Итог
# Чтобы создать модель лучшего качества, надо подобрать конфигурацию, которая даст лучшее качество.
#
# Пример подсчета качества для лучшей конфигурации с урока:

# X_text получен из датасета, однако при желании, его можно преобразовать перед векторизацией

# векторизация через CountVectorizer без параметров:
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(X_text)  # вектора примеров

# классификация через LogisticRegression без параметров:

# scores = []
#
# for _ in range(100):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#     clf = LogisticRegression()
#     clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     scores.append(score)
#
# print(sum(scores) / 100)
