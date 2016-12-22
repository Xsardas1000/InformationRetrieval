from urllib import error
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import defaultdict
import scipy as sp
import pymorphy2
import re
import requests
import math
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


def prepare_text(text):

    blob = TextBlob(str(text))
    docs = [str(sentence) for sentence in blob.sentences]
    stemmed_docs = [stem_file(doc) for doc in docs]
    return stemmed_docs, docs

def get_text_from_url(url):
    try:
        page = requests.get(url)
    except error.HTTPError as err:
        if err.code != 200:
            print("Error while loading page with code ", err.code)
        else:
            raise
    soup = BeautifulSoup(page.text, "lxml")
    try:
        text = soup.findAll('p')
    except IndexError:
        text = ["Unknown"]

    parts = [p.text for p in text]
    return ' '.join(parts)

def cos_raw(v1, v2):
    return sp.spatial.distance.cosine(v1.toarray(), v2.toarray())

def range_search(request, corpus):
    distances = []
    for i, doc in enumerate(corpus):
        vec = corpus.getrow(i)
        distances.append((cos_raw(vec, request), i))
    return distances

def print_doc(request, doc):
    splitted_doc = doc.split(" ")
    splitted_doc = [word for word in splitted_doc if len(word) > 0]
    highlighted_doc = ""

    for word in splitted_doc:
        stemmed_word = stem_file(word)
        if len(stemmed_word) < 1:
            highlighted_doc += word + " "
            continue
        if stemmed_word[0] in request:
            highlighted_doc += "\033[31;48m " + word + " \033[m"
        else:
            highlighted_doc += word + " "
    print(highlighted_doc)
    print("\n")

def print_highlighted(request, docs, distances, num):
    if num > len(distances): num = len(distances)
    for weight, index in distances[:num]:
        print("Weight: ", 1.0 - weight)
        print("Doc_number: ", index)
        print_doc(request, docs[index])

def search_sentence(url, idf, value, IDCG_value, request):
    text_from_url = get_text_from_url(url)

    prepared_sentences, docs = prepare_text(text_from_url)
    if idf == False:
        print("Search without idf")
        vectorizer = TfidfVectorizer(stop_words=stopwords.words("russian"), use_idf=False, norm='l2')
    else:
        print("Search with idf")
        vectorizer = TfidfVectorizer(stop_words=stopwords.words("russian"), use_idf=True, norm='l2')

    corpus = vectorizer.fit_transform([' '.join(item) for item in prepared_sentences])
    request = stem_file(request)
    key_words = request
    request = vectorizer.transform([' '.join(request)])
    distances = range_search(request, corpus)
    distances = sorted(distances, key=lambda item: item[0])
    '''print_highlighted(key_words, docs, distances, num=len(docs))'''

    values_list = []
    for weight, index in distances:
        if str(index) in value:
            values_list.append(value[str(index)])
        else:
            values_list.append(0)
    print(values_list)
    print("DCG = ", dcg(values_list))
    print("IDCG = ", dcg(IDCG_value))
    print("NDCG = ", dcg(values_list) / dcg(IDCG_value))



def dcg(scale_values):
    if len(scale_values) < 1:
        return 0
    DCG = scale_values[0]
    for i, scale in enumerate(scale_values[1:]):
        DCG += scale / math.log2(i + 2)
    return DCG



def search(doc_number, idf):
    urls = ["https://ru.wikipedia.org/wiki/Лорд-распорядитель",
            "https://ru.wikipedia.org/wiki/Шлайфстайн,_Джозеф",
            "https://ru.wikipedia.org/wiki/История_почты_и_почтовых_марок_Люксембурга"]

    requests = ["Должность первого высшего сановника Великобритании вакантна уже почти 600 лет.",
                "Узник Бухенвальда к моменту освобождения первую половину жизни провёл в гетто, а вторую — в концлагере.",
                "Почтовую марку с надписью «Слава СССР» эмитировали в Великом Герцогстве Люксембург."]

    values = ([defaultdict(int, {"1": 2, "2": 1, "4": 1, "7": 1, "8": 2, "12": 2}),
           defaultdict(int, {"3": 2, "9": 1, "12": 2, "16": 1, "44": 1, "26": 1, "38": 1}),
           defaultdict(int, {"17":2, "26":2, "28":1, "43":2, "22":2, "33":2, "2":1, "25":2, "32":2, "15":1, "49":1, "10":1, "44":1,
                             "18":1, "9":1, "24":1, "38":1})])
    IDCG_values = [[2, 2, 2, 1, 1, 1],
               [2, 2, 1, 1, 1, 1, 1],
               [2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1 ]]

    print("Request:", requests[doc_number])
    search_sentence(urls[doc_number], idf, values[doc_number], IDCG_values[doc_number], requests[doc_number])

if __name__ == '__main__':
    search(doc_number=2, idf=False)
