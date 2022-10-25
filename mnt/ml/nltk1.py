import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer # It has the ability to lemmatize.
import tensorflow as tensorF # A multidimensional array of elements is represented by this symbol.
from tensorflow.keras import Sequential # Sequential groups a linear stack of layers into a tf.keras.Model
from tensorflow.keras.layers import Dense, Dropout
import sys

# Uncomment this if needed
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download('omw-1.4')

class DitaAjaNLTK():
    def __init__(self):
        self.tagList = []
        self.patternList = []
        self.documentX = []
        self.documentY = []
        self.model = None
        self.lm = None
        self.originalFaq = {}

    def nltk_init(self):
        self.lm = WordNetLemmatizer() #for getting words
        # lists
        self.tagList = []
        self.patternList = []
        self.documentX = []
        self.documentY = []


        with open('./faq.json') as file:
            self.originalFaq = json.load(file)

        for intent in self.originalFaq:
            if "patterns" in intent:
                for pattern in intent["patterns"]:
                    newToken = nltk.word_tokenize(pattern.lower())# tokenize the patterns
                    self.patternList.extend(newToken)# extends the tokens
                    self.documentX.append(pattern.lower())
                    self.documentY.append(intent["tag"])
            if intent["tag"] not in self.tagList:# add unexisting tags to their respective classes
                self.tagList.append(intent["tag"]) 
        self.patternList = [self.lm.lemmatize(word.lower()) for word in self.patternList if word not in string.punctuation] # set words to lowercase if not in punctuation
        self.patternList = sorted(set(self.patternList))# sorting words
        self.tagList = sorted(set(self.tagList))# sorting classes

        dataTraining = [] # training list array
        outEmpty = [0] * len(self.tagList)
        # bow(Bag Of Words) model
        for idx, doc in enumerate(self.documentX):
            bagOfwords = []
            text = self.lm.lemmatize(doc.lower())
            for word in self.patternList:
                bagOfwords.append(1) if word in text else bagOfwords.append(0)

            outputRow = list(outEmpty)
            outputRow[self.tagList.index(self.documentY[idx])] = 1
            dataTraining.append([bagOfwords, outputRow])

        print(dataTraining)
        random.shuffle(dataTraining)
        print("")
        print(dataTraining)
        print("")
        dataTraining = num.array(dataTraining, dtype=object)# coverting our data into an array after shuffling
        print(dataTraining)
        # print(dataTraining)

        x = num.array(list(dataTraining[:, 0]))# first training phase
        y = num.array(list(dataTraining[:, 1]))# second training phase
        iShape = (len(x[0]),)
        oShape = len(y[0])
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
        # Output the model in summary
        # Whilst training your Neural Network, you have the option of making the output verbose or simple.
        self.model.fit(x, y, epochs=100, verbose=1)
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
        print(bagOfWords)
        return num.array(bagOfWords)

    def pred_class(self, text, vocab, labels): 
        bagOfWords = self.wordBag(text, vocab)
        result = self.model.predict(num.array([bagOfWords]))[0]
        # print("TESTING")
        # print("")
        # print(result)
        threshold = 0.5
        yp = [[idx, res] for idx, res in enumerate(result) if res > threshold]


        yp.sort(key=lambda x: x[1], reverse=True)
        newList = []
        for r in yp:
            newList.append(labels[r[0]])
        return newList

    def getRes(self, firstlist, fJson): 
        tag = firstlist[0]
        listOfIntents = fJson
        for i in listOfIntents: 
            if i["tag"] == tag:
                response = random.choice(i["responses"])
                break
        return {
            "tag": tag,
            "response": response
        }
    
    def getResponse(self, text):
        text = text.lower()
        intents = self.pred_class(text, self.patternList, self.tagList)
        result = self.getRes(intents, self.originalFaq)
        return result