
# coding: utf-8

# In[ ]:


#Importing necessary libraries and opening text files
import math as m
import random
import time
import sys
readFile1 =  open('amazon_cells_labelled.txt', 'r')
readFile2 =  open('imdb_labelled.txt', 'r')
readFile3 =  open('yelp_labelled.txt', 'r')


# In[ ]:


'''
The Data Handler class is responsible for the training and testing of the Naive Bayes Classifier.
It takes .txt files and trains on 80 percent of the data in the file.
From this, it calculates likelihoods of a given word in a certain class and probabilities of the occurence of a class.
The remaining 20 percent of the data will be used for testing.
Creating an instance of this class calculates log(likelihoods) for each word given a specific class and stores it in a
hashmap for referencing.
'''
class DataHandler():
    
    #Initializing constructor of the data handler class
    def __init__(self,*files):
        self.files= files
        self.trainingData , self.testData = self.splitToData()
        self.negative, self.positive, self.classes, self.freqLookup = self.splitToBags()
        self.vocabulary = self.createVocabulary()
        self.priors = self.calculateLogPriors()
        self.logLikelihoods = self.calculateLogLikelihoods()
        self.test = self.test()
        self.accuracy = self.testAccuracy()
        self.summary = self.printSummary()
        

    #Function to split into training and test data
    def splitToData(self):
        data =[]
        for file in self.files:
            for line in file:
                cleanLine = line.rstrip('\n').split('\t')
                sentence = cleanLine[0].replace(".","").replace("!","").replace("-","").replace(":","").replace(")","").replace(";","").replace("@","").replace("(","").replace(",","").replace("&","").replace('"','').replace("?","").replace("*","").replace("+","").lower()
                features = sentence.split(' ')
                tag = cleanLine[1]
                pair = (features,tag)
                data.append(pair)
        file.close()
        random.shuffle(data)
        trainingData= data[0:int(0.8*len(data))]
        testData = data[int(0.8*len(data)):]
        return(trainingData, testData)

   # Function to split training data into positive and negative bags of words and a list to keep all classes.
   # Here, the frequencies of each word in each class is calculated and stored in a dictionary.

    def splitToBags(self):
        positive =[]
        negative =[]
        classes = []
        freqLookup ={}
        for doc in self.trainingData:
            classes.append(doc[1])
            if doc[1]== '1':
                for word in doc[0]:
                    if (word.lower(),1) not in freqLookup:
                        freqLookup[(word.lower(),1)] = 1
                    else:
                        freqLookup[(word.lower(),1)]+= 1 
                    positive.append(word.lower())
                    if (word.lower(),0) not in freqLookup:
                        freqLookup[(word.lower(),0)] = 0
            elif doc[1]== '0':
                for word in doc[0]:
                    if (word,0) not in freqLookup:
                        freqLookup[(word.lower(),0)] = 1
                    else:
                        freqLookup[(word.lower(),0)]+= 1
                    if (word.lower(),1) not in freqLookup:
                        freqLookup[(word.lower(),1)] = 0
                    negative.append(word.lower())
        return(negative,positive,classes,freqLookup)

    #Fuction that loops through positive and negative words to create vocabulary of unique words
    def createVocabulary(self):
        vocabulary =[]
        for word in self.negative:
            if word not in vocabulary:
                vocabulary.append(word)
        for word in self.positive:
            if word not in vocabulary:
                vocabulary.append(word)
        return vocabulary

    # Function to calculate for the prior probability of the classes
    def calculateLogPriors(self):
        negClass =[]
        priors = {}
        for i in self.classes:
            if i == '0':
                negClass.append(1)
        negProb = sum(negClass)/len(self.classes)
        posProb = 1-negProb
        negPrior = m.log10(negProb)
        posPrior = m.log10(posProb)
        priors['0'] = negPrior
        priors['1'] = posPrior
        return priors
    
    #Function to find likelihood of a word
    def calculateLogLikelihoods(self):
        likelihoodLookup ={}
        posDenominator =[]
        negDenominator =[]
        for word in self.vocabulary:
            posDenominator.append(self.freqLookup[(word,1)]+1)
            negDenominator.append(self.freqLookup[(word,0)]+1)
        for word in self.vocabulary:
            posLikelihood = m.log10((self.freqLookup[(word,1)]+1)/sum(posDenominator))
            likelihoodLookup[(word,1)] = posLikelihood
            negLikelihood = m.log10((self.freqLookup[(word,0)]+1)/sum(negDenominator))
            likelihoodLookup[(word,0)] = negLikelihood
        return likelihoodLookup
    
    #Function to test data
    def test(self):
        result = {}
        summary =[]
        for testDoc in self.testData:
            testSentence = (' ').join(testDoc[0])
            sumPosLikelihood = self.priors['1']
            sumNegLikelihood = self.priors['0']
            for i in testDoc[0]:
                if i in self.vocabulary:
                    sumPosLikelihood+= self.logLikelihoods[(i,1)]
                    sumNegLikelihood+= self.logLikelihoods[(i,0)]
                else:
                    sumPosLikelihood+= 0
                    sumNegLikelihood+= 0
            result['0'] = sumNegLikelihood
            result['1'] = sumPosLikelihood
            if result['0']>result['1']:
                summary.append((testSentence,'0'))
            else:
                summary.append((testSentence,'1'))
        return summary
    
    #Calulating accuracy
    def testAccuracy(self):
        correct = []
        for i in range (len(self.test)):
            if self.test[i][1]== self.testData[i][1]:
                correct.append(1)
            else:
                pass
        return(round((sum(correct)/len(self.test))*100,2))
    
    #Printing data handler summary
    def printSummary(self):
        print("--------------------------")
        print("Length of training data: ", len(self.trainingData))
        print("Length of test data: ", len(self.testData))
        print("Words in positive bag: ", len(self.positive))
        print("Words in negative bag: ", len(self.negative))
        print("Total word count: ", (len(self.positive)+len(self.negative)))
        print("Unique words in vocabulary: ", len(self.vocabulary))
        print("Log(prob) of negative class: ", self.priors['0'])
        print("Log(prob) of positive class:", self.priors['1'])
        print("Test Accuracy: ", self.accuracy, "%")
        print("--------------------------")

                      


# In[ ]:


'''
The NaiveBayesClassifier class takes a data handler and a file to classify.
An instance of this class runs through the documents in a file and predicts the class they belong to.
'''
class NaiveBayesClassifier():
    def __init__(self,handler,file):
        self.handler = handler
        self.file = open(str(file),'r')
        self.data = self.splitToData()
        self.classify = self.classify()
        
    def splitToData(self):
        data =[]
        for line in self.file:
            cleanLine = line.rstrip('\n').split('\t')
            sentence = cleanLine[0].replace(".","").replace("!","").replace("-","").replace(":","").replace(")","").replace(";","").replace("@","").replace("(","").replace(",","").replace("&","").replace('"','').replace("?","").replace("*","").replace("+","").lower()
            features = sentence.split(' ')
            tag = cleanLine[1]
            pair = (features,tag)
            data.append(pair)
        self.file.close()
        return data
    
    def classify(self):
        result={}
        summary =[]
        for testDoc in self.data:
            testSentence = (' ').join(testDoc[0])
            sumPosLikelihood = self.handler.priors['1']
            sumNegLikelihood = self.handler.priors['0']
            for i in testDoc[0]:
                if i in self.handler.vocabulary:
                    sumPosLikelihood+= self.handler.logLikelihoods[(i,1)]
                    sumNegLikelihood+= self.handler.logLikelihoods[(i,0)]
            result['0'] = sumNegLikelihood
            result['1'] = sumPosLikelihood
            if result['0']>result['1']:
                summary.append((testSentence,'0'))
            else:
                summary.append((testSentence,'1'))
        for i in summary:
            print(i,'\n')
        return summary
            
    def predictionAccuracy(self):
        correct = []
        for i in range (len(self.classify)):
            if self.classify[i][1]== self.data[i][1]:
                correct.append(1)
            else:
                pass
        return(round((sum(correct)/len(self.classify))*100,2))
    
        


# In[ ]:


#Testing Naive Bayes Classifier
def main(argv):
    start = time.time()
    handler = DataHandler(readFile1,readFile2,readFile3)
    classifier = NaiveBayesClassifier(handler,argv[1])
    print("Prediction accuracy: ", classifier.predictionAccuracy(), "%\n")
    end = time.time()
    print("Completed in "+str(round((end-start),2))+' seconds')
    
main(sys.argv)

