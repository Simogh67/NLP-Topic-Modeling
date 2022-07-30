# NLP-Topic-Modeling

In this repository, we perform topic modeling by using Latent Dirichlet Allocation (LDA), which is used to classify text in a document to a particular topic. 
It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. 

* Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words.
* LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial. 
* It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution. 

### Dataset
 
OpinRank Review Dataset Data Set is used in this repository [https://archive.ics.uci.edu/ml/datasets/opinrank+review+dataset]. The data set contains user reviews of cars and hotels collected from Tripadvisor (259,000 reviews) and Edmunds (42,230 reviews).

An instance of hotel reviews is "Peaceful and comfortable four star hotel. The Anting Villa is a very good hotel, set amid well kept gardens on a quiet road in a leafy suburb of Shanghai. My room was comfortable, modern, and very well equipped; a comfortable bed, easy chair and footstool, and a bathroom with full bath and plenty of space for toiletries."

Also, an instance of car reviews is "Buy this car with your eyes wide open and your expectations in line - you'll be happy. Reading the previous reviews for the RDX, I wonder whether some simply chose the wrong car. Don't buy a race horse, and then complain that it eats too much or can't pull a loaded wagon. This turbocharged engine has its quirks, but drive it right and you'll be rewarded with decent mileage and good acceleration."

We use part of the dataset (Dubai hotel reviews and cars reviews in 2009).

First, we mix some hotel review files with some car review files. Then, LDA is used for topic modeling.

Finally, to evaluate our model, we feed two unseen reviews (a car review plus a hotel review).

Our model classifies the unseen data i.e. predicts the unseen data belongs to which topic.

### Data Preprocessing 

We will perform the following steps:

* **Tokenization**: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
* Words that have fewer than 3 characters are removed.
* All **stopwords** are removed.
* Words are **lemmatized** - words in third person are changed to first person and verbs in past and future tenses are changed into present.
* Words are **stemmed** - words are reduced to their root form.

### Result

We set the number of topics to equal to ten. Here is the result of LDA model:

Topic: 0 
Words: 0.010*"fantast" + 0.009*"return" + 0.008*"amaz" + 0.007*"visit" + 0.007*"experi" + 0.007*"year" + 0.007*"holiday" + 0.006*"wonder" + 0.006*"club" + 0.006*"drink"


Topic: 1 
Words: 0.010*"water" + 0.008*"drink" + 0.007*"fresh" + 0.007*"build" + 0.007*"fruit" + 0.006*"bathroom" + 0.006*"bottl" + 0.006*"towel" + 0.005*"floor" + 0.005*"sensor"


Topic: 2 
Words: 0.021*"climat" + 0.018*"glove" + 0.018*"hill" + 0.015*"pedal" + 0.011*"satellit" + 0.011*"stabl" + 0.011*"watch" + 0.011*"ignit" + 0.009*"desert" + 0.008*"cadillac"


Topic: 3 
Words: 0.014*"taxi" + 0.011*"shop" + 0.011*"busi" + 0.010*"mall" + 0.010*"walk" + 0.010*"airport" + 0.010*"citi" + 0.008*"valu" + 0.008*"minut" + 0.007*"free"


Topic: 4 
Words: 0.023*"control" + 0.021*"radio" + 0.019*"sound" + 0.018*"rear" + 0.017*"wheel" + 0.017*"power" + 0.016*"stereo" + 0.015*"bluetooth" + 0.015*"featur" + 0.014*"light"


Topic: 5 
Words: 0.012*"problem" + 0.010*"tell" + 0.009*"check" + 0.009*"say" + 0.008*"know" + 0.007*"leav" + 0.007*"thing" + 0.006*"review" + 0.006*"start" + 0.006*"tri"


Topic: 6 
Words: 0.146*"fuel" + 0.101*"economi" + 0.053*"perform" + 0.047*"qualiti" + 0.043*"style" + 0.032*"effici" + 0.028*"design" + 0.026*"tech" + 0.023*"quiet" + 0.023*"build"


Topic: 7 
Words: 0.017*"apart" + 0.014*"floor" + 0.014*"small" + 0.012*"bathroom" + 0.011*"wash" + 0.011*"kitchen" + 0.010*"famili" + 0.010*"bedroom" + 0.010*"sleep" + 0.010*"okay"


Topic: 8 
Words: 0.014*"steer" + 0.012*"test" + 0.012*"buy" + 0.010*"vehicl" + 0.010*"engin" + 0.010*"car" + 0.010*"toyota" + 0.009*"featur" + 0.009*"power" + 0.009*"purchas"


Topic: 9 
Words: 0.022*"mile" + 0.019*"mileag" + 0.016*"highway" + 0.016*"road" + 0.013*"vehicl" + 0.013*"get" + 0.012*"averag" + 0.011*"trip" + 0.011*"citi" + 0.009*"model"

As we can see, we have well-diversified topics. For instance, clearly topics zero and two are about hotels, but topics 
one and five are about cars. Interestingly, since we use Dubai hotel reviews, topic nine indicates that many positive reviews were about Burj al Arab hotel and its luxury facilities. This implies that our model could extract meaningful information from the dataset. 

### Evaluation

Here, we have an unseen review i.e. "small room and tiny en-suite but friendly and helpful Location is excellent - 4 minute walk from Earls Court tube station. Receptionists were friendly and helpful - got us to our room at 11am and looked after our suitcase after checkout while we went to the city". This text clearly is about a hotel room.

Our model predicts that with 72% chance, the text belongs to topic zero. The result is correct and shows that our model works.

The next unseen review is about a car "I researched this car for a few months and decided on the Buick  rather than a small Lexus or Mercedes. The new JDE Power dependability ratings make me feel really good about my selection. Cannot say anything bad about the car. Lively in tight situations while still exhibiting the Buick trademark smooth ride. Great fit and finish. Very pleased."

Our model predicts that with 83% chance, the text belongs to topic five. The result is the closest answer we have among topics and shows the merit of our model.
