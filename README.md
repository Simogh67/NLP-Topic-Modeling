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
Words: 0.011*"taxi" + 0.010*"busi" + 0.009*"shop" + 0.009*"mall" + 0.009*"walk" + 0.008*"citi" + 0.008*"airport" + 0.008*"small" + 0.006*"free" + 0.006*"valu"


Topic: 1 
Words: 0.030*"fuel" + 0.016*"dealer" + 0.014*"problem" + 0.010*"economi" + 0.009*"tell" + 0.008*"say" + 0.007*"issu" + 0.006*"know" + 0.006*"nois" + 0.006*"leav"


Topic: 2 
Words: 0.017*"hill" + 0.017*"bathroom" + 0.014*"shower" + 0.012*"fresh" + 0.011*"fruit" + 0.011*"stabl" + 0.010*"reclin" + 0.010*"bath" + 0.009*"separ" + 0.008*"cold"


Topic: 3 
Words: 0.022*"kid" + 0.013*"glove" + 0.012*"adult" + 0.010*"park" + 0.010*"famili" + 0.010*"children" + 0.009*"peopl" + 0.009*"watch" + 0.008*"expens" + 0.008*"lot"


Topic: 4 
Words: 0.018*"control" + 0.016*"wheel" + 0.016*"power" + 0.016*"rear" + 0.015*"steer" + 0.015*"sound" + 0.015*"radio" + 0.014*"style" + 0.013*"featur" + 0.012*"exterior"


Topic: 5 
Words: 0.014*"mile" + 0.014*"mileag" + 0.013*"vehicl" + 0.012*"buy" + 0.010*"purchas" + 0.010*"road" + 0.010*"get" + 0.009*"highway" + 0.009*"power" + 0.008*"engin"


Topic: 6 
Words: 0.163*"stereo" + 0.049*"dual" + 0.029*"minivan" + 0.019*"chrome" + 0.016*"telescop" + 0.015*"listen" + 0.013*"roof" + 0.013*"batteri" + 0.012*"cute" + 0.011*"music"


Topic: 7 
Words: 0.010*"return" + 0.009*"drink" + 0.008*"fantast" + 0.007*"even" + 0.007*"visit" + 0.007*"holiday" + 0.006*"check" + 0.006*"upgrad" + 0.006*"club" + 0.006*"worth"


Topic: 8 
Words: 0.044*"brake" + 0.025*"review" + 0.023*"read" + 0.016*"experi" + 0.014*"peopl" + 0.014*"qualiti" + 0.013*"star" + 0.012*"rat" + 0.011*"amaz" + 0.010*"world"


Topic: 9 
Words: 0.022*"moon" + 0.014*"sleek" + 0.014*"burj" + 0.014*"sensor" + 0.014*"mina" + 0.013*"beauti" + 0.013*"golf" + 0.012*"fantast" + 0.011*"amaz" + 0.010*"luxuri"

As we can see, we have well-diversified topics. For instance, clearly topics zero and two are about hotels, but topics 
one and five are about cars. Interestingly, since we use Dubai hotel reviews, topic nine indicates that many positive reviews were about Burj al Arab hotel and its luxury facilities. This implies that our model could extract meaningful information from the dataset. 

### Evaluation

Here, we have an unseen review i.e. "small room and tiny en-suite but friendly and helpful Location is excellent - 4 minute walk from Earls Court tube station. Receptionists were friendly and helpful - got us to our room at 11am and looked after our suitcase after checkout while we went to the city". This text clearly is about a hotel room.

Our model predicts that with 72% chance, the text belongs to topic zero. The result is correct and shows that our model works.

The next unseen review is about a car "I researched this car for a few months and decided on the Buick  rather than a small Lexus or Mercedes. The new JDE Power dependability ratings make me feel really good about my selection. Cannot say anything bad about the car. Lively in tight situations while still exhibiting the Buick trademark smooth ride. Great fit and finish. Very pleased."

Our model predicts that with 83% chance, the text belongs to topic five. The result is the closest answer we have among topics and shows the merit of our model.
