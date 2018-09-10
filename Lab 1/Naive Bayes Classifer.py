import math as m

document = []
classes = []

#Calculating for the probability of 
def calculateLogPriors(classes,c):
    count = 0
    for i in classes:
        if i == c:
            count+= 1
        else:
            pass
    priorProb = count/len(classes)
    return (m.log10(priorProb))
