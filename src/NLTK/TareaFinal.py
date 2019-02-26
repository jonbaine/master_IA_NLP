import nltk
#For the os.walk
import os
import re
import csv

#For Plotting
import matplotlib.pyplot as plt
#For interpolate
import scipy.interpolate

from nltk.probability import LidstoneProbDist, LaplaceProbDist, WittenBellProbDist, SimpleGoodTuringProbDist
from nltk.model import NgramModel

#libmagic for pdf detection
import magic

#PDF handling
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

#command handling
import subprocess

def extractContent (target):
    content = ''

    with open(target, 'r') as content_file:
        content = content_file.read()
    return content


def extractContentFromPDF (target):
    """ Extract the content in text from a PDF file."""
    try:
        input_ = file(target, 'rb')
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        process_pdf(manager, converter, input_)
    except:
        print 'File ' + target + ' Not allowed to mine the PDF! \o/'
        return ''
    return output.getvalue() 

def writeContent (target, content):
    with open(target, 'w')as write_file:
        write_file.write(content)

def cleanTextFromLynxStuff (content):
    #The nltk.clean_html is not necessary, the lynx has cleaned the whole HTML tag.
    #We should make the removal of the parts inserted by the lynx.
    retValue = ''
    contentAfterBrakRemoval = re.sub('\[.*?\]','',content)
    #Remove the references section
    if 'Referencias' in contentAfterBrakRemoval:
        retValue = contentAfterBrakRemoval.split('Referencias')[0]
    else:
        retValue = contentAfterBrakRemoval
    return retValue



def contains_entity_names(_line, _target):

    try:
        if _target in _line:
            #print _line
            return True

    except:
        print 'Exception!'
    
    return False

#Ideas:
# stremming (utilizar las raices de las palabras en lugar de la palabra literal)
# utilizar idiomas, aunque parece que todos los resultados se estan dando en ingles.
#Para el stemming podemos probar el PorterStemmer. o lematizacion de algun tipo.

def createCorpus(target, _ner = False):
    print 'Create the corpus for: ' + str(target)
    corpus = []
    #init the libmagic.
    ms = magic.open(magic.MAGIC_NONE)
    ms.load()
    targetName = target.split('/')[-1:][0]
    print targetName
    for root, directories, files in os.walk(target):
        for fileName in files:
            
            contentAfterCleaning=''
            fileTarget = root+'/'+fileName
            filetype = ms.file(fileTarget)
            #Epty file!
            if filetype == None:
                continue
            elif 'PDF' in filetype:
                #handle the PDF!
                contentAfterCleaning = extractContentFromPDF (fileTarget)
            else:
                content = extractContent(fileTarget)
                contentAfterCleaning = cleanTextFromLynxStuff (content)
            #Tokenize in sentences. We should care if there is other language appart from engligh. 
            #The english is used by default!.
            sentences = nltk.sent_tokenize(contentAfterCleaning)
            
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                #Convert all tolower...
                words = map(lambda word:word.lower(),words)
                #Remove the non alphanum stuff
                words = filter(lambda word: word.isalnum(), words)
                if _ner :
                    if  contains_entity_names(words, targetName):
                        corpus.append(words)
                else:
                    corpus.append(words)

    return corpus




class Tweet:
    """ Class to store the Tweet information"""
    subset=''
    entity=''
    tweet_num=0
    tweet_id=''
    tweet_content=''
    label=''
    num_related = 0
    num_unrelated = 0
    num_undecidable = 0
    reconciled = False
    rating = 0.0
    def  __init__(self, _subset, _entity, _tweet_num, _tweet_id, _tweet_content, _label, _num_related, _num_unrelated, _num_undecidable, _reconciled):
        self.subset = _subset
        self.entity = _entity
        self.tweet_num = int(_tweet_num)
        self.tweet_id = _tweet_id
        self.tweet_content=_tweet_content
        self.label=_label
        self.num_related = int(_num_related)
        self.num_unrelated = int(_num_unrelated)
        self.num_undecidable = int(_num_undecidable)
        self.reconciled = bool(_reconciled)

    def getRelatedVerdict(self):
        if self.label =='related':
            return True
        else:
            return False

    def clean(self):
        #steps to clean the unnecessary stuff from the tuits...
        #Should the hashtags (#) and users (@) be removed? 
        self.tweet_content = re.sub('\#.*? ','',self.tweet_content)
        self.tweet_content = re.sub('\@.*? ','',self.tweet_content)
        #Remove URLS
        self.tweet_content = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',self.tweet_content)
        wordsOfTweet = nltk.word_tokenize(self.tweet_content)
        #Remove the non alphanum stuff
        wordOfTweetWithoutAlpNum = filter(lambda word: word.isalnum(), wordsOfTweet)
        self.tweet_content = ' '.join(map(lambda word:word.lower(),wordOfTweetWithoutAlpNum))
        

    def rate(self, model, _numberOfNGram):
        #normalize perplexity by length
        if len(self.tweet_content)>0:
            self.rating = model.perplexity(self.tweet_content.split(' '))/len(self.tweet_content)


def loadTweets(target):
    retValue = []
    with open ('./CORPUS/TWEETS/'+target+'-tweets.tsv') as tsvFile:
        tsvreader = csv.reader (tsvFile, delimiter = '\t')
        removeHeader = True
        for row in tsvreader:

            #Avoid the 1st line.
            if removeHeader:
                removeHeader = False
                continue

            readed_tuit = Tweet(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
            if len(readed_tuit.tweet_content)>0:
                retValue.append(readed_tuit)
    return retValue

def calculateInterpolatedPrecision (_pointsToInterpolate,  _recall, _precision):
    retValue = []
    #Zip the points.
    points = zip (_recall, _precision)
    parts = _pointsToInterpolate
    start_end_pairs = zip(parts[:-1], parts[1:])
    start_end_pairs_index = 0    
    actual_start_end = start_end_pairs[start_end_pairs_index]
    maxPrecisionValue = 0.0
    for p in points:

        if p[1] >= actual_start_end[0] and p[1] <= actual_start_end[1]:
            if maxPrecisionValue < p[0]:
                maxPrecisionValue = p[0]
        else:
            retValue.append(maxPrecisionValue)
            maxPrecisionValue = 0.0
            start_end_pairs_index +=1 
            actual_start_end = start_end_pairs[start_end_pairs_index]

    retValue.append(maxPrecisionValue)


    return retValue
            
        
def calculateMAPValue(tweets, precisionAcumulate):
    tweet_len = len(tweets)
    #All the tweets...
    numberOfRelevantTweets = 0
    acumulatedPrecision = 0.0
    for tweet, preci  in zip (tweets, precisionAcumulate):
        if tweet.getRelatedVerdict():
            acumulatedPrecision += preci
            numberOfRelevantTweets += 1
    return acumulatedPrecision/numberOfRelevantTweets


class EvaluationResult:
    interpolatedRecall=[]
    interpolatedPrecision=[]
    mapValues = 0.0
    
    def  __init__(self, _recall, _precision, _mapvalues):
        self.interpolatedRecall=_recall
        self.interpolatedPrecision=_precision
        self.mapValues = _mapvalues



def evaluateCharacterization (tweets):
    """This function should evaluate with differente metrics how the tweets has been characterized"""
    # The metrics are -> precission, recall and MAP.
    # precision = |R|/|T| , where |T| are the total ranked documents, and |R| are the relevants ones.
    # Recall = |R|/|U| , where the |U| are the relevant stuff for this query.
    # Those characterizations, are not valid for ranked stuff.
    # Print a recall curve plotting precission (p) as a function of recall (r)
    # Ave(P) = sum(P(k)*rel(k))/number_of_relevant_documents
    #Pero vamos a imprimir lo que viene siendo P en funcion de recall:
    # aveP = 1/11 sum( p interp (r)) r E {0, 0.1, 0.2, ..., 1.0}
    # p interpolado (r) -> max p (i) i where i >= r
    numberOfRelatedTweets = sum(1 for tweet in tweets if tweet.getRelatedVerdict() == True )
    precision = 1.0
    recall = 0.0
    precisionAcumulate = []
    recallAcumulate=[]
    accumulatedRelatedTweets = 0
    analizedTweet = 0
    for tweet in tweets:
        analizedTweet+=1
        if tweet.getRelatedVerdict():
            accumulatedRelatedTweets+=1
        precision = float(accumulatedRelatedTweets)/analizedTweet
        recall = float(accumulatedRelatedTweets)/numberOfRelatedTweets
        precisionAcumulate.append(precision)
        recallAcumulate.append(recall)

    interpolatedRecall = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    interpolatedPrecision  = [1.0] + calculateInterpolatedPrecision(interpolatedRecall, precisionAcumulate, recallAcumulate )

    return EvaluationResult(interpolatedRecall, interpolatedPrecision, calculateMAPValue(tweets, precisionAcumulate))
    



def processSource (corpus, tweets, _numberOfNGram):

    # estimator for smoothing the N-gram model LAPLACE  
    estimatorLindstone = lambda fdist, bins: LidstoneProbDist(fdist, 0.1)  
    estimatorLaplace = lambda fdist, bins: LaplaceProbDist(fdist)

    model = NgramModel(_numberOfNGram, corpus, False, True, estimatorLindstone )


    #Clean the stuff...
    for tweet in tweets:
        tweet.clean()
        tweet.rate(model, _numberOfNGram)


    #remove the empty tweets from the list (if after the clean they were just hashtags or links.)
    tweets = filter(lambda T: len(T.tweet_content)>0, tweets)
    #SORT IT!
    tweets.sort(key=lambda T: T.rating, reverse=False)

    return tweets

def printValues (_name, _values):
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision VS Recall -'+ str(_name))

    
    plt.axis([0.0, 1.0, 0.0, 1.0])
    labels = []

    index = 1
    for value in _values:

        plt.plot(value.interpolatedRecall, value.interpolatedPrecision, marker='o')
        print  'N-Gram ' + str(index) + ' MAP ' + str(value.mapValues)
        labels.append('N-Gram ' + str(index))
        index+= 1
        
    plt.legend(labels, fancybox=True,shadow=True, loc=3)
    plt.savefig(_name+'.png', bbox_inches='tight')
    plt.close()



    

def main ():
    targets = ['bart','bayer','blockbuster','boingo','cadillac','fender','harpers','luxor','mgm','rover']
    for target in targets:
        corpus = createCorpus ('./CORPUS/DOCUMENTS/'+target, _ner=True)
            
        tweets = loadTweets (target)

        returnedValues = []
        for i in xrange(1,6):
            returnedValues.append(evaluateCharacterization(processSource(corpus, tweets, i)))
        printValues(target, returnedValues)

    return
            

if __name__ == "__main__":
   main()
