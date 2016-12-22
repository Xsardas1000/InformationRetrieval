from nltk.corpus import stopwords
import pymorphy2
import re
import os, sys, fnmatch
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def stem_file(text):
    text = re.sub(r"(\n)", " ", text.lower())
    text = re.split("[^а-я0-9]", text)
    morph = pymorphy2.MorphAnalyzer()
    stemmed_text = []
    for word in text:
        if len(word) > 0:
            stemmed_text.append(morph.parse(word)[0].normal_form)
    stemmed_text = [word for word in stemmed_text if word not in stopwords.words("russian")]
    return stemmed_text

def find_files(source_path, mask):
    find_files = []
    for root, dirs, files in os.walk(source_path):
        find_files += [os.path.join(root, name) for name in files if fnmatch.fnmatch(name, mask)]
    return find_files

def make_vectorizer(processed_files):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_files)
    return vectorizer, X


def get_files(source_paths):
    source_files = []
    for path in source_paths:
        with open(path, 'r') as file:
            source_files.append(file.read())
    return source_files

def binary_classification(class1_paths, class2_paths, class1_name, class2_name):

    files1 = get_files(class1_paths)
    files2 = get_files(class2_paths)

    processed_files_1 = np.array([stem_file(file) for file in files1])
    processed_files_2 = np.array([stem_file(file) for file in files2])
    processed_files = np.concatenate([processed_files_1, processed_files_2])
    processed_files = [' '.join(file) for file in processed_files]

    y_train = np.concatenate([len(processed_files_1) * [0], len(processed_files_2) * [1]])
    vectorizer, X = make_vectorizer(processed_files)
    X_train = np.array(X.toarray())

    model = MultinomialNB()
    model.fit(X_train, y_train)

    while 1:
        test_file_name = input("Enter test file name: ")
        target = input("Enter the class of test file: ")
        with open(test_file_name, 'r') as file:
            test_text = file.read()
            test_stemmed = stem_file(test_text)
            test_vec = vectorizer.transform([' '.join(test_stemmed)])
            X_test = np.array(test_vec.toarray())

            y_test = target
            expected = [y_test]
            predicted = model.predict(X_test)
            print(model.predict_proba(X_test))
            if predicted[0] == 0:
                print(class1_name)
            elif predicted[0] == 1:
                print(class2_name)




if __name__ == '__main__':
    class1_paths = find_files('./Sport', '*.txt')
    class2_paths = find_files('./Science', '*.txt')


    binary_classification(class1_paths, class2_paths, "Sport", "Science")
