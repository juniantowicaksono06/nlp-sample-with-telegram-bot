import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer # It has the ability to lemmatize.
import tensorflow as tensorF # A multidimensional array of elements is represented by this symbol.
from tensorflow.keras import Sequential # Sequential groups a linear stack of layers into a tf.keras.Model
from tensorflow.keras.layers import Dense, Dropout
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sys
from nltk.corpus import stopwords
import string
import re
from collections import Counter
from os import path
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords

# Uncomment this if needed
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download('omw-1.4')

class DitaAjaNLP():
    def __init__(self):
        self.articleID = []
        self.tagList = []
        self.documentX = []
        self.documentY = []
        self.model = None
        self.lm = None
        self.desmitaFaq = {}
        self.kataDasar = []
        self.stop_words = stopwords.words('indonesian')
        self.nltk_kata_dasar = self.stop_words
        self.stop_words = str(self.stop_words)


        self.kataDasar.extend(self.kataDasar)
        self.kataDasar.sort()
        self.kataDasar = str(self.kataDasar)
        with open('./kata_dasar.txt', 'r') as file:
            self.kataDasar = file.read().splitlines()
        self.kataDasar = str(self.kataDasar)
        self.WORDS = Counter(self.words(self.kataDasar))

    def tensorflow_train(self, all_data):
        model_path = 'ml_model/ditaaja_model.h5'
        self.lm = WordNetLemmatizer() #for getting words
        # lists
        self.articleID = []
        self.tagList = []
        self.documentX = []
        self.documentY = []


        self.desmitaFaq = all_data 
        if 'article_id' in self.desmitaFaq:
            article_id = self.desmitaFaq['article_id']
            article_title = self.desmitaFaq['article_title']
            article_tag = self.desmitaFaq['article_tag']
            for (index, id) in enumerate(article_id):
                tags = article_tag[index]
                if len(tags) > 0:
                    for tag in tags:
                        if tag not in self.documentX:
                            token = nltk.word_tokenize(tag.lower())
                            self.tagList.extend(token)
                            self.documentX.append(tag.lower())
                            # self.documentY.append(article_id[index])
                            self.documentY.append(article_id[index])
                if article_id[index] not in self.articleID:
                    self.articleID.append(article_id[index])
        # print(self.documentX, flush=True)
        self.tagList = sorted(set(self.tagList))
        self.articleID = sorted(set(self.articleID))

        dataTraining = [] # training list array
        outEmpty = [0] * len(self.articleID)
        # bow(Bag Of Words) model
        for idx, doc in enumerate(self.documentX):
            bagOfwords = []
            text = self.lm.lemmatize(doc.lower())
            for word in self.tagList:
                bagOfwords.append(1) if word in text else bagOfwords.append(0)

            outputRow = list(outEmpty)
            outputRow[self.articleID.index(self.documentY[idx])] = 1
            dataTraining.append([bagOfwords, outputRow])

        random.shuffle(dataTraining)
        # print(dataTraining, flush=True)
        dataTraining = num.array(dataTraining, dtype=object)# coverting our data into an array after shuffling

        x = num.array(list(dataTraining[:, 0]))# first training phase
        y = num.array(list(dataTraining[:, 1]))# second training phase
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        iShape = (len(x_train[0]),)
        print(iShape, flush=True)
        oShape = len(y_train[0])
        if not path.isfile(model_path):
            # parameter definition
            self.model = Sequential()
            # In the case of a simple stack of layers, a Sequential model is appropriate

            # Dense function adds an output layer
            self.model.add(Dense(128, input_shape=iShape, activation="relu"))
            # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
            self.model.add(Dropout(0.5))
            # Dropout is used to enhance visual perception of input neurons
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(oShape, activation = "softmax"))
            # below is a callable that returns the value to be used with no arguments
            md = tensorF.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
            # Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
            self.model.compile(loss='categorical_crossentropy',
                        optimizer=md,
                        metrics=["accuracy"])
        else:
            # self.model.load_model(model_path)
            self.model = None
            self.model = load_model(model_path)
            print(self.model.summary(), flush=True)
        print(x_train.shape, flush=True)
        print(y_train.shape, flush=True)
        self.model.fit(x_train, y_train, epochs=100, verbose=0, validation_data=(x_test, y_test))
        self.model.save(model_path)
        # By epochs, we mean the number of times you repeat a training set.

    def tokenizeWord(self, text): 
        newtkns = nltk.word_tokenize(text)
        newtkns = [self.lm.lemmatize(word) for word in newtkns]
        return newtkns

    def wordBag(self, text, vocab): 
        newtkns = self.tokenizeWord(text)
        bagOfWords = [0] * len(vocab)
        for w in newtkns: 
            for idx, word in enumerate(vocab):
                if word == w: 
                    bagOfWords[idx] = 1
        # print(bagOfWords)
        return num.array(bagOfWords)

    def pred_class(self, text, vocab, articleID): 
        bagOfWords = self.wordBag(text, vocab)
        result = self.model.predict(num.array([bagOfWords]))[0]
        print("Hasil Prediksi:", flush=True)
        print(result, flush=True)
        print(articleID, flush=True)
        # print("TESTING")
        # print("")
        # print(result)
        threshold = 0.5
        yp = [[idx, res] for idx, res in enumerate(result) if res > threshold]


        yp.sort(key=lambda x: x[1], reverse=True)
        newList = []
        for r in yp:
            newList.append(articleID[r[0]])
        return newList

    def getRes(self, firstlist, fJson): 
        articleID = firstlist[0]
        faq = fJson
        # print(faq)
        print(articleID)
        # article_id = []
        for (index, value) in enumerate(faq['article_id']): 
            if value == articleID:
                response = faq['article_title'][index]
                break
        return {
            "articleID": articleID,
            "response": response
        }
    
    def directResponse(self, text):
        for (index, faq) in enumerate(self.desmitaFaq['article_title']):
            if faq == text:
                return {
                    "articleID": self.desmitaFaq['article_id'][index],
                    "response": faq
                }
        return False
    
    def getResponse(self, text):
        text = text.lower()
        text = self.removePunctuationAndWS(text)
        directResponse = self.directResponse(text)
        print("HASIL", flush=True)
        print(directResponse, flush=True)
        if directResponse != False:
            return directResponse
        intents = self.pred_class(text, self.tagList, self.articleID)
        result = self.getRes(intents, self.desmitaFaq)
        return result
    
    def removePunctuationAndWS(self, text):
        sentence = re.sub(r"\s+", " ", text)
        sentence = sentence.strip()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        return sentence
    
    def getTextTag(self, text):
        # lowercase_sentence = text.lower()
        sentence = text.lower()
        greetings = []
        if path.isfile('./greetings.json'):
            with open('./greetings.json') as file:
                greetings = json.load(file)
        # Detect kalimat pembuka
        if len(greetings) > 0 and len(sentence.split(' ')) >= 2:
            greeting_level = 3 if len(sentence.split(' ')) > 2 else 2
            index = 0
            text_tmp = sentence.split(' ')
            while index < greeting_level:
                if text_tmp[index].strip() in greetings:
                    text_tmp.pop(index)
                    index = 0
                    continue
                index += 1
            sentence = " ".join(text_tmp).strip()
        elif sentence.strip() in greetings:
            sentence = ""

        new_words = []        
        if sentence != "":
            sentence = self.removePunctuationAndWS(sentence)
            # Tokenizing
            token = nltk.word_tokenize(sentence)

            new_words = [self.correction(word) for word in token if self.correction(word) not in self.stop_words]
        return new_words

        # ### TOKENIZING
        # # REMOVE ANGKA
        # lowercase_sentence = re.sub(r"\d+", "", lowercase_sentence)
        # # REMOVE PUNCTUATION
        # lowercase_sentence = lowercase_sentence.translate(str.maketrans("","", string.punctuation))
        # # REMOVE WHITESPACE LEADING & TRAILING
        # lowercase_sentence = lowercase_sentence.strip()
        # # REMOVE MULTIPLE WHITESPACE INTO SINGLE WHITESPACE
        # lowercase_sentence = re.sub(r"\s+", " ", lowercase_sentence)
        # # TOKENIZE
        # self.lm = WordNetLemmatizer() #for getting words
        # token = self.tokenizeWord(lowercase_sentence)
        # # token = word_tokenize(lowercase_sentence)

        # factory = StemmerFactory()
        # stemmer = factory.create_stemmer()
        # output = []
        # for word in token:
        #     output.append(stemmer.stem(word))
        # list_words = str(stopwords.words('indonesian'))
        # token_without_stopword = [ word for word in output if not word in list_words ]
        # return token_without_stopword
    
    def P(self, word):
        # "Probability of `word`."
        N=sum(self.WORDS.values())
        return self.WORDS[word] / N

    def correction(self, word):
        # "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        # "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        # "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        # "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)] # [('', 'kemarin'), ('k', 'emarin'), ('ke', 'marin'), dst]
        deletes    = [L + R[1:]               for L, R in splits if R] # ['emarin', 'kmarin', 'kearin', dst]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1] # ['ekmarin', 'kmearin', 'keamrin', dst]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters] # ['aemarin', 'bemarin', 'cemarin', dst]
        inserts    = [L + c + R               for L, R in splits for c in letters] # ['akemarin', 'bkemarin', 'ckemarin', dst]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        # "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    def words(self, text): 
        return re.findall(r'\w+', text.lower())