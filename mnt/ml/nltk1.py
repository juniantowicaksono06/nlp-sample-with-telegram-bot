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
        self.ourClasses = []
        self.newWords = []
        self.documentX = []
        self.documentY = []
        self.ourNewModel = None
        self.lm = None
        self.ourData = {}

    def nltk_init(self):
        self.lm = WordNetLemmatizer() #for getting words
        # lists
        self.ourClasses = []
        self.newWords = []
        self.documentX = []
        self.documentY = []

        self.ourData = {"intents": [
            
             {"tag": "age",
              "patterns": ["how old are you?"],
              "responses": ["I am 21 years old and my birthday was yesterday", "I am 21 years old right now"]
             },
              {"tag": "greeting",
              "patterns": [ "Hi", "Hello", "Hey"],
              "responses": ["Hi there", "Hello", "Hi :)"],
             },
              {"tag": "goodbye",
              "patterns": [ "bye", "later", "goodbye"],
              "responses": ["Bye", "take care"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "who are you?"],
              "responses": ["I have no name yet", "You can give me one and i will apreciate"]
             },
             {
                "tag": "hobby",
                "patterns": ["What is your hobby?", "Hobby?"],
                "responses": ["My hobbies are gaming and coding", "Gaming and coding, sometimes i like to watch movie too :)"]
             },
             {
                "tag": "skill",
                "patterns": ["What's your skill?", "skill", "Expert"],
                "responses": ["Coding, Gaming, Linux, Etc", "Love to play with code and linux :)"]
             },
             {
                "tag": "compliment",
                "patterns": ["Nice", "good", "terrific", "good bot"],
                "responses": ["Thank you for your generous compliment", "Thanks", "Thanks a lot"]
             }
            
        ]}

        # print(self.ourData)
        for intent in self.ourData['intents']:
            for pattern in intent["patterns"]:
                print(pattern)
                ournewTkns = nltk.word_tokenize(pattern.lower())# tokenize the patterns
                self.newWords.extend(ournewTkns)# extends the tokens
                self.documentX.append(pattern.lower())
                self.documentY.append(intent["tag"])
            if intent["tag"] not in self.ourClasses:# add unexisting tags to their respective classes
                self.ourClasses.append(intent["tag"]) 
        # print(self.newWords)
        self.newWords = [self.lm.lemmatize(word.lower()) for word in self.newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
        # print(self.newWords)
        self.newWords = sorted(set(self.newWords))# sorting words
        self.ourClasses = sorted(set(self.ourClasses))# sorting classes

        trainingData = [] # training list array
        outEmpty = [0] * len(self.ourClasses)
        # bow model
        for idx, doc in enumerate(self.documentX):
            bagOfwords = []
            text = self.lm.lemmatize(doc.lower())
            for word in self.newWords:
                bagOfwords.append(1) if word in text else bagOfwords.append(0)

            outputRow = list(outEmpty)
            outputRow[self.ourClasses.index(self.documentY[idx])] = 1
            trainingData.append([bagOfwords, outputRow])

        random.shuffle(trainingData)
        trainingData = num.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

        x = num.array(list(trainingData[:, 0]))# first trainig phase
        y = num.array(list(trainingData[:, 1]))# second training phase
        iShape = (len(x[0]),)
        oShape = len(y[0])
        # parameter definition
        self.ourNewModel = Sequential()
        # In the case of a simple stack of layers, a Sequential model is appropriate

        # Dense function adds an output layer
        self.ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
        # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
        self.ourNewModel.add(Dropout(0.5))
        # Dropout is used to enhance visual perception of input neurons
        self.ourNewModel.add(Dense(64, activation="relu"))
        self.ourNewModel.add(Dropout(0.3))
        self.ourNewModel.add(Dense(oShape, activation = "softmax"))
        # below is a callable that returns the value to be used with no arguments
        md = tensorF.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        # Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
        self.ourNewModel.compile(loss='categorical_crossentropy',
                    optimizer=md,
                    metrics=["accuracy"])
        # Output the model in summary
        # Whilst training your Nural Network, you have the option of making the output verbose or simple.
        self.ourNewModel.fit(x, y, epochs=100, verbose=1)
        # By epochs, we mean the number of times you repeat a training set.

    def ourText(self, text): 
        newtkns = nltk.word_tokenize(text)
        newtkns = [self.lm.lemmatize(word) for word in newtkns]
        return newtkns

    def wordBag(self, text, vocab): 
        newtkns = self.ourText(text)
        bagOwords = [0] * len(vocab)
        for w in newtkns: 
            for idx, word in enumerate(vocab):
                if word == w: 
                    bagOwords[idx] = 1
        return num.array(bagOwords)

    def pred_class(self, text, vocab, labels): 
        bagOwords = self.wordBag(text, vocab)
        ourResult = self.ourNewModel.predict(num.array([bagOwords]))[0]
        newThresh = 0.2
        yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

        yp.sort(key=lambda x: x[1], reverse=True)
        newList = []
        for r in yp:
            newList.append(labels[r[0]])
        return newList

    def getRes(self, firstlist, fJson): 
        tag = firstlist[0]
        listOfIntents = fJson["intents"]
        for i in listOfIntents: 
            if i["tag"] == tag:
                ourResult = random.choice(i["responses"])
                break
        return ourResult
    
    def getResponse(self, text):
        intents = self.pred_class(text, self.newWords, self.ourClasses)
        ourResult = self.getRes(intents, self.ourData)
        return ourResult