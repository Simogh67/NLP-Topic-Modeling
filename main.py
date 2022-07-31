# import libraries
import os
import glob
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from operator import itemgetter

# inputs
stemmer = SnowballStemmer("english")
stop = STOPWORDS
file_location_hotel = os.path.join('data', 'hotel', '*')
file_location_car = os.path.join('data', 'car', '*')
file_location_unseen_hotel = os.path.join('data', 'unseen_hotel', '*')
file_location_unseen_car = os.path.join('data', 'unseen_car', '*')
v_hotel = [0, 1, 3, 5, 7]  # contains the topics that describe hotels data

# required functions


def get_data(file_location, c_car):
    filenames = glob.glob(file_location)
    data = []
    for file in filenames:
        with open(file) as f:
            while (line := f.readline().rstrip()):
                if c_car == 1:
                    if '<TEXT>' in line or '<FAVORITE>' in line:
                        data.append(line)
                else:
                    data.append(line)
    return data


def cleaner(data):
    data = [re.sub(r"[^a-zA-Z0-9]", " ", d.lower()) for d in data]
    return data


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def stemmer_lemmetizer(texts):

    d_clean = [[word for word in simple_preprocess(str(doc)) if word not in
                stop and len(word) > 3] for doc in texts]
    d_clean = [list(set([lemmatize_stemming(word) for word in simple_preprocess
                         (str(doc))]))
               for doc in d_clean]
    return d_clean


def unseen_data_tokenizer(data, dictionary):

    doc_clean = [word for word in simple_preprocess(str(data)) if word not in
                 stop and len(word) > 3]
    doc_clean = [list(set([lemmatize_stemming(word)
                      for word in simple_preprocess(str(doc_clean))]))]
    bow_vector = [dictionary.doc2bow(doc) for doc in doc_clean]
    return bow_vector


def get_lda_result(model):
    for idx, topic in model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")


def get_prediction_result(prediction, name):
    for topic in prediction:
        print('result of prediction {}: {}'.format(name, topic))


def get_result(prediction):
    result = []
    for topic in prediction:
        result.append(topic)
    return result[0]


def classifier(text, dictionary, model, v_hotel):
    bow_vector = unseen_data_tokenizer(text, dictionary)
    prediction = model[bow_vector]
    result = get_result(prediction)
    topic = max(result, key=itemgetter(0))[0]
    if topic in v_hotel:
        c_class = 1
    else:
        c_class = 0
    return c_class


def get_classification(text, dictionary, model, v_hotel):
    result = []
    for t in text:
        r = classifier(t, dictionary, model, v_hotel)
        result.append(r)
    return result


def get_accurcy(y_pred_hotel, y_pred_car):
    numerator = y_pred_hotel.count(1) + y_pred_car.count(0)
    denominator = y_pred_hotel.count(
        1) + y_pred_car.count(0) + y_pred_hotel.count(0)
    +y_pred_car.count(1)
    acc = numerator / denominator
    return acc


def main():

    # getting hotel and car data
    data_hotel = get_data(file_location_hotel, c_car=0)
    data_car = get_data(file_location_car, c_car=1)
    data = data_hotel + data_car

    # cleaning the data
    data = cleaner(data)
    data = stemmer_lemmetizer(data)

    # building dictionary and bag-of-words model
    dictionary = corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]

    # LDA model
    model = LdaModel(doc_term_matrix, num_topics=10, id2word=dictionary,
                     passes=10)

    # getting the result of LDA model
    get_lda_result(model)

    # getting the unseen hotel data
    df_unseen_hotel = get_data(file_location_unseen_hotel, c_car=0)
    df_unseen_hotel = cleaner(df_unseen_hotel)
    y_pred_hotel = get_classification(
        df_unseen_hotel, dictionary, model, v_hotel)

    # getting the unseen car data
    df_unseen_car = get_data(file_location_unseen_car, c_car=1)
    df_unseen_car = cleaner(df_unseen_car)
    y_pred_car = get_classification(df_unseen_car, dictionary, model, v_hotel)

    # accurcy
    acc = get_accurcy(y_pred_hotel, y_pred_car)
    print('the accurcy of the model:{}'.format(acc))


if __name__ == "__main__":
    main()
