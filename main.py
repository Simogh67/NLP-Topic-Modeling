# import libraries 
import os
import glob
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models.ldamodel import LdaModel
from gensim import corpora

# inputs
stemmer = SnowballStemmer("english")
stop=STOPWORDS
file_location_hotel = os.path.join('data', 'hotel','*')
file_location_car = os.path.join('data', 'car','*')

# required functions

def get_data(file_location,c_car):
    filenames = glob.glob(file_location)
    data=[]
    for file in filenames:
        with open(file) as f:
            while (line := f.readline().rstrip()):
                if c_car==1:
                 if '<TEXT>' in line or '<FAVORITE>' in line:
                   data.append(line)
                else:
                    data.append(line)
    return data

def cleaner(data):
    data=[re.sub(r"[^a-zA-Z0-9]", " ", d.lower()) for d in data]
    return data

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def stemmer_lemmetizer(texts):
    
    d_clean=[[word for word in simple_preprocess(str(doc)) if word not in 
                  stop and len(word)>3] for doc in texts]
    d_clean=[list(set([lemmatize_stemming(word) for word in simple_preprocess
                       (str(doc))]))
              for doc in d_clean]
    return d_clean

def unseen_data_tokenizer(data,dictionary):
    
    doc_clean=[word for word in simple_preprocess(str(data)) if word not in 
                  stop and len(word)>3]
    doc_clean=[list(set([lemmatize_stemming(word) for word in simple_preprocess
                        (str(doc_clean))]))]
    bow_vector =  [dictionary.doc2bow(doc) for doc in doc_clean]
    return bow_vector

def get_lda_result(model):
    for idx, topic in model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")

def get_prediction_result(prediction,name):
    for topic in prediction:
        print('result of prediction {}: {}'.format(name,topic))
        
def main():
 
 # getting hotel and car data 
 data_hotel = get_data(file_location_hotel,c_car=0)
 data_car = get_data(file_location_car,c_car=1)
 data= data_hotel+data_car
 # cleaning the data
 data=cleaner(data)  
 data=stemmer_lemmetizer(data)
 # building dictionary and bag-of-words model
 dictionary = corpora.Dictionary(data)
 dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)
 doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]
 # LDA model
 model =  LdaModel (doc_term_matrix, num_topics=10, id2word = dictionary, 
                       passes=10)
 # getting the result of LDA model
 get_lda_result(model)
 # getting the unseen data 
 unseen_hotel = ["small room and tiny en-suite but friendly and helpful \
                   Location is excellent - 4 minute walk from Earls \
                   Court tube station. Receptionists were friendly and \
                   helpful - got us to our room at 11am and looked after \
                   our suitcase after checkout while we went to the city."] 
                   
 unseen_car=["I researched this car for a few months and decided on the Buick \
            rather than a small Lexus or Mercedes. The new JDE Power \
            dependability ratings make me feel really good about my selection.\
            Cannot say anything bad about the car. Lively in tight situations \
            while still exhibiting the Buick trademark smooth ride. Great fit \
            and finish. Very pleased."]
 # hotel prediction result           
 bow_vector=unseen_data_tokenizer(unseen_hotel,dictionary)                  
 prediction = model[bow_vector]
 get_prediction_result(prediction,'hotel') 
 # car prediction result  
 bow_vector=unseen_data_tokenizer(unseen_car,dictionary)                  
 prediction = model[bow_vector]
 get_prediction_result(prediction,'car') 
if __name__ == "__main__":
    main()