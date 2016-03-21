#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
import nltk
from nltk.corpus import stopwords
from pyspark import SparkContext
from pyspark import Broadcast
from pyspark.accumulators import AccumulatorParam
import sys
import re
import math
import matplotlib.pyplot as plt


INPUT_FILE_PATTERN = '^"(.+)","(.+)",(.*),(.*),(.*)'
MAPPING_FILE_PATTERN = '^(.+),(.+)'
STOPWORDS = stopwords.words('english')

# Pre-bin counts of false positives for different threshold ranges
BINS = 101
nthresholds = 100

###########################################################################
#functions to read the source and target input files into RDD objects  ####

def parseInputLine(line):

    match = re.search(INPUT_FILE_PATTERN, line)

    if match is None:
        print('Invalid datafile line: {0}'.format(line))
        return (line, -1)
    elif match.group(1) == 'id':
        print('Header datafile line: {0}'.format(line))
        return (line, 0)
    else:
        product = '%s %s %s' % (match.group(2), match.group(3), match.group(4))
    return ((match.group(1), product), 1)


def tokenize(string):
    return  filter(None, re.split(r'\W+', string.lower()))


def filterStopWords(string):
    
    tokenizedString = tokenize(string)
    filteredString = []
        
    for word in tokenizedString:
        if not word in STOPWORDS:
            filteredString.append(word)
    return filteredString


def readFile(string):

    lines = sc.textFile(string)
    inputLines = lines.map(parseInputLine).cache()
    pairRDD = inputLines.filter(lambda x: x[1] == 1).map(lambda x: x[0])

    return pairRDD.mapValues(filterStopWords)

################################################################################
#functions to read the gold standard mapping file into RDD object ##############

def parseMappingLine(line):
    
    match = re.search(MAPPING_FILE_PATTERN, line)
    if match is None:
        print('Invalid datafile line: {0}'.format(line))
        return (line, -1)
    elif match.group(1) == 'id1':
        print('Header datafile line: {0}'.format(line))
        return (line, 0)
    else:
        #key = '%s %s' % (match.group(1), match.group(2))
        id1 = '%s' % (match.group(1))
        id2 = '%s' % (match.group(2))
        key = (id1, id2)
        return ((key, 'gold'), 1)


def readMappingFile(string):
    
    lines = sc.textFile(string)
    inputLines = lines.map(parseMappingLine).cache()
    pairRDD = inputLines.filter(lambda x: x[1] == 1).map(lambda x: x[0])
    
    return pairRDD


################################################################################
#functions to calculate term frequency and inverse document frequency ##########

#implementation of the tf function
def tf(string):

    tfDict = {}
    for word in string:
        if tfDict.has_key(word):
            tfDict[word] = tfDict[word]+1
        else:
            tfDict[word] = 1

    for k, v in tfDict.iteritems():
        tfDict[k] = float(v)/len(string)

    return tfDict

#implementation of the idf function
def idf(corpusRDD):
    N = float(corpusRDD.count())
    flatMapCorpus = corpusRDD.flatMap(lambda record: set(record[1]))
    tokenCountOneCorpus = flatMapCorpus.map(lambda token: (token, 1))
    tokenCountCorpus = tokenCountOneCorpus.reduceByKey(lambda a,b : a+b)
    return tokenCountCorpus.map(lambda token: (token[0], N/token[1])).sortBy(lambda x: x[1])


#implementation of tfidf funciton
def tfidf(tokens, idfs):
    tfs = tf(tokens)
    tfIdfDict = {token: (tfs[token] * idfs[token]) for token in tfs}
    return tfIdfDict

###################################################################################
#functions to calculate cosine similarity between records##########################
def dotProd(a,b):
    return sum(a[key]*b[key] for key in a.keys() if key in b)

def norm(a):
    return math.sqrt(dotProd(a, a))

def invert(record):
    tokens = record[1]
    pairs = [(token, record[0])for token in tokens]
    return (pairs)

def fastCosineSimilarity(record, amazonNormsBroadcast, googleNormsBroadcast, amazonWeightsRDDBroadcast, googleWeightsRDDBroadcast):
    ids = record[0]
    amazonRec = ids[0]
    googleRec = ids[1]
    tokens = record[1]
    value = record[1]
  
    s = sum(amazonWeightsRDDBroadcast.value[amazonRec][token]*googleWeightsRDDBroadcast.value[googleRec][token] for token in tokens)
    value = float(s)/(googleNormsBroadcast.value[googleRec]*amazonNormsBroadcast.value[amazonRec])
    
    key = (amazonRec, googleRec)
    return (key, value)

##########################################################################################
#made only minor modifications to this part of code that was already given in the tutorial
#all this code was given in the tutorial...probably because none of it deals with
#the pyspark transformations in spark.

def gs_value(record):
    if (record[1][1] is None):
        return 0
    else:
        return record[1][1]

class VectorAccumulatorParam(AccumulatorParam):
    # Initialize the VectorAccumulator to 0
    def zero(self, value):
        return [0] * len(value)
    
    # Add two VectorAccumulator variables
    def addInPlace(self, val1, val2):
        for i in xrange(len(val1)):
            val1[i] += val2[i]
        return val1


#Return a list with entry x set to value and all other entries set to 0
def set_bit(x, value, length):
    bits = []
    for y in xrange(length):
        if (x == y):
            bits.append(value)
        else:
            bits.append(0)
    return bits


def bin(similarity):
    return int(similarity * nthresholds)


def add_element(score):
    global fpCounts
    b = bin(score)
    fpCounts += set_bit(b, 1, BINS)

# Remove true positives from FP counts
def sub_element(score):
    global fpCounts
    b = bin(score)
    fpCounts += set_bit(b, -1, BINS)


def falsepos(threshold):
    fpList = fpCounts.value
    return sum([fpList[b] for b in range(0, BINS) if float(b) / nthresholds >= threshold])

def falseneg(threshold, trueDupSimsRDD):
    return trueDupSimsRDD.filter(lambda x: x < threshold).count()

def truepos(threshold, trueDupSimsRDD, falsenegDict):
    return trueDupSimsRDD.count() - falsenegDict[threshold]


# Precision = true-positives / (true-positives + false-positives)
# Recall = true-positives / (true-positives + false-negatives)
# F-measure = 2 x Recall x Precision / (Recall + Precision)

def precision(threshold, trueposDict, falseposDict):
    tp = trueposDict[threshold]
    return float(tp) / (tp + falseposDict[threshold])

def recall(threshold, trueposDict, falsenegDict):
    tp = trueposDict[threshold]
    return float(tp) / (tp + falsenegDict[threshold])

def fmeasure(threshold, trueposDict, falsenegDict, falseposDict):
    r = recall(threshold, trueposDict, falsenegDict)
    p = precision(threshold, trueposDict, falseposDict)
    return 2 * r * p / (r + p)



#################### main function #########################################
def main(sc):
    

    fullGooglePairRDD = readFile(sys.argv[1])
    fullAmazonPairRDD = readFile(sys.argv[2])
    goldStandardRDD = readMappingFile(sys.argv[3])
    
    print('\nTotal number of records in the google dataset = {0}'.format(fullGooglePairRDD.count()))
    print('\nTotal number of records in the amazon dataset = {0}'.format(fullAmazonPairRDD.count()))
    print('\nTotal number of records in the goldStandard dataset = {0}'.format(goldStandardRDD.count()))
    
    
    fullCorpusRDD = fullAmazonPairRDD.union(fullGooglePairRDD)
    fullIdfWeights = idf(fullCorpusRDD)
    print('There are %s unique tokens combined in both the datasets.' % fullIdfWeights.count())
    

    #getting idf weights for tokens in both the datasets
    idfFullWeights = fullIdfWeights.collectAsMap()
  
    #compute and keep the combined tf-idf weights for tokens in all records in both datasets
    amazonWeightsRDD = fullAmazonPairRDD.map(lambda x: (x[0], tfidf(x[1], idfFullWeights)))
    googleWeightsRDD = fullGooglePairRDD.map(lambda x: (x[0], tfidf(x[1], idfFullWeights)))
    
    #calculate norm for all records in both the datasets to be used while calculating dot product
    amazonNorms = amazonWeightsRDD.map(lambda x: (x[0], norm(x[1])))
    googleNorms = googleWeightsRDD.map(lambda x: (x[0], norm(x[1])))
    
    """commonTokens is a very important datastructure. Apply pyspark transformations to keep only the common tokens between
    each pair of records from both the datasets
    This way we get rid of quadratic complexity incurred by calculating dot product between each pair of records from both the datasets """
    amazonInvPairsRDD = amazonWeightsRDD.flatMap(invert)
    googleInvPairsRDD = googleWeightsRDD.flatMap(invert)
    commonTokens = (amazonInvPairsRDD.join(googleInvPairsRDD).map(lambda x: (x[1], x[0])).groupByKey())
  
    #brodcast the big datastructures that will be used frequently to the worker nodes for efficiency
    amazonNormsBroadcast = sc.broadcast(amazonNorms.collectAsMap())
    googleNormsBroadcast = sc.broadcast(googleNorms.collectAsMap())
    amazonWeightsRDDBroadcast = sc.broadcast(amazonWeightsRDD.collectAsMap())
    googleWeightsRDDBroadcast = sc.broadcast(googleWeightsRDD.collectAsMap())

    #calculate cosine similarity for each pair in commonTokens
    similaritiesFull = commonTokens.map(lambda x: fastCosineSimilarity(x, amazonNormsBroadcast, googleNormsBroadcast, amazonWeightsRDDBroadcast, googleWeightsRDDBroadcast))
    
    #join the cosine similarities between the two datasets  with gold standard mapping file
    trueDupSimsRDD = (goldStandardRDD.leftOuterJoin(similaritiesFull).map(gs_value).cache())

    #all the code below deals with getting an optimal value of "cosine similarity threshold" for this dataset
    simsFullValuesRDD = (similaritiesFull.map(lambda x: x[1]).cache())
    
    #plotting the graphs to get the threshold cosine similarity by statistical measures
    zeros = [0] * BINS
    global fpCounts
    fpCounts = sc.accumulator(zeros, VectorAccumulatorParam())
    simsFullValuesRDD.foreach(add_element)
    trueDupSimsRDD.foreach(sub_element)
    
    thresholds = [float(n) / nthresholds for n in range(0, nthresholds)]
    falseposDict = dict([(t, falsepos(t)) for t in thresholds])
    falsenegDict = dict([(t, falseneg(t, trueDupSimsRDD)) for t in thresholds])
    trueposDict = dict([(t, truepos(t, trueDupSimsRDD, falsenegDict)) for t in thresholds])
    precisions = [precision(t, trueposDict, falseposDict) for t in thresholds]
    recalls = [recall(t, trueposDict, falsenegDict) for t in thresholds]
    fmeasures = [fmeasure(t, trueposDict, falsenegDict, falseposDict) for t in thresholds]
   
    
    fig = plt.figure()
    plt.plot(thresholds, precisions)
    plt.plot(thresholds, recalls)
    plt.plot(thresholds, fmeasures)
    plt.legend(['Precision', 'Recall', 'F-measure'])
    
    fig.savefig("./figure.png")
    
    sc.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: readInputFile <file>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="ER")



#################call to the main function#########################################
main(sc)
