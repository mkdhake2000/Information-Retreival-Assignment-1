#Import pickle library to read pickle files created in Preprocessing
import pickle as pkl

#Loading the termFreq.pkl file to get the term frequency
fl=open("21111405-ir-systems/pklFiles/termFreq.pkl", "rb")
termFreq = pkl.load(fl)

#Loading the docFreq.pkl file to get the document frequency
fl=open("21111405-ir-systems/pklFiles/docFreq.pkl", "rb")
docFreq = pkl.load(fl)

#Loading the docLenName.pkl file to get the length and names of files
fl=open("21111405-ir-systems/pklFiles/docLenName.pkl", "rb")
docLenName = pkl.load(fl)

#Loading the fileWords.pkl file to get the words after tokenization from each file
fl=open("21111405-ir-systems/pklFiles/fileWords.pkl", "rb")
fileWords = pkl.load(fl)

#Loading the invDocFreq.pkl file to get the inverse document frequency
fl=open("21111405-ir-systems/pklFiles/invDocFreq.pkl", "rb")
invDocFreq = pkl.load(fl)

#Getting the list of names of all files
fileNames = list(docLenName.keys())
#Getting the list of all unique tokens
uniqueTokens = list(set(termFreq.keys()))

#Importing num2words library to convert numbers into words
from num2words import num2words as n2w
#Importing sys library to get the name of query file as argument
import sys
#Importing re library to get ASCII characters from file by using regex from this library
import re
#Importing math library 
import math
#Importing itemgetter from operator library to get items from dictionary
from operator import itemgetter as ig
#Importing pandas library to export results in csv files
import pandas as pd
#Importing word_tokenize from nltk.tokenize library to tokenize the file contents
from nltk.tokenize import word_tokenize as tokenizer
#Importing Stemmer from nltk.stem library for stemming the file contents
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
#Importing stopwords from nltk.corpus to get a list of stopwords to remove from file contents
from nltk.corpus import stopwords as sW
#Getting the stopWords list for English
StopWords = set(sW.words('english'))
#Creating PorterStemmer() object
stemmer=PorterStemmer()
#Getting Stop_words list of stopwords without "and" and "or" as they are used as connectors in query
Stop_words = [i for i in StopWords if (i != "and") and (i != "or")]


# name of file containing queries
inName = sys.argv[1]
# adding extension to the file
inName=inName+'.txt'
# Opening the file
queryFile=open(inName,"r")

#Dictionary containing queries as keys and query ids as values
queryDict = {}

#For loop to get all queries from the query file
for q in queryFile:
    queryText = q.split("\t")
    queryDict[queryText[1].replace("\n","")] = queryText[0].replace("\n", "")
queryList = list(queryDict.keys())

#Boolean Retrieval System
def brs():
    

    #docNameList contains list of list of names of documents with relevance to queries
    docNameList = {}
    #relevList contains list of list of query relevance to documents
    relevList = {}
    #For loop to initialize docNameList and relevList
    for k,v in queryDict.items():
        docNameList[k]=[]
        relevList[k]=[]


    #For loop to process all queries of the query file
    for query,qID in queryDict.items():
        #Remove HTML tags
        htmlTags = re.compile('<.*?>')
        queryText = re.sub(htmlTags, '', query)
        #Remove punctuation marks
        queryText = re.sub(r'[^\w\s]',' ',queryText)
        #Remove non-ASCII characters
        encodedQuery = queryText.encode("ascii","ignore")
        queryText=encodedQuery.decode()
        #Tokenization of the stream
        queryTokens = tokenizer(queryText)
        #Converting number tokens to words
        queryTokenNo=[]
        for j in queryTokens:
            if j.isdigit() and len(j)<4:
                j=n2w(int(j))
            queryTokenNo.append(j)

        queryTokens=queryTokenNo
        #Converting tokens into lower-case
        queryTokens = [token.lower() for token in queryTokens]
        #Stemming tokens
        queryTokens = [stemmer.stem(token) for token in queryTokens]
        #Removing stopwords from tokens except "and" and "or"
        queryTokens = [token for token in queryTokens if token not in Stop_words]
        tokens=queryTokens

        #queryTokens contains the tokens with connectors attached appropriately
        queryTokens = []
        #query to be processed only if it's length is greater than 0
        if len(tokens)>0:
            queryTokens = [tokens[0]]

        #Storing connectors in a seperate list
        conn = ["and","or"]

        #For loop to add connectors in query. If available already, we don't add explicitly
        for i in range(1,len(tokens)):
            if tokens[i] in conn:
                if queryTokens[-1] in conn:
                    continue
            else:
                if queryTokens[-1] not in conn:
                    queryTokens.append("and")
            queryTokens.append(tokens[i])

        #connectors contains all connectors from query
        connectors = []
        #queryWords contains all other words except connectors from query
        queryWords = []
        #For loop to seperate query into connectors list and queryWords list
        for token in queryTokens:
            if token.lower() in conn:
                connectors.append(token.lower())
            else:
                queryWords.append(token)

        #noFiles contains no. of Files
        noFiles = len(fileNames)
        #tokenPres dictionary tells about presence of a word from query in the files
        tokenPres = {}
        #For loop to fill the tokenPres dictionary
        for token in range(len(queryWords)):
            tokPres = [0]*noFiles
            if queryWords[token] in uniqueTokens:
                for docNo in termFreq[queryWords[token]].keys():
                    tokPres[docNo] = 1
            tokenPres[token] = tokPres

        #Storing first element of tokenPres to result to process it
        result = {i:tokenPres[0][i] for i in range(len(tokenPres[0]))}
        #For loop to check for connectors between words in a query and appropriately performing actions to get the result as files which contain the query words
        for cn in range(len(connectors)):
            if connectors[cn] == "and":
                result.update({i:(result[i] & tokenPres[cn+1][i]) for i in range(len(result))})
            if connectors[cn] == "or":
                result.update({i:(result[i] | tokenPres[cn+1][i]) for i in range(len(result))})
            
        #Sorting dictionary as items according to value in descending order
        result= dict(sorted(result.items(), key=ig(1), reverse=True))

        #For loop to get the top documents with relevance>0 for all queries and their relevance(0 or 1)
        for i in range(20):
            docNameList[query].append(fileNames[(list(result.keys()))[i]].split(".")[0])
            if result[(list(result.keys()))[i]]==0:
                relevList[query].append(0)
            else:
                relevList[query].append(1)
            
    #Creating lists of query id, iteration, doc id, relevance to output in csv files
    queryId =  [ele for ele in list(queryDict.values()) for i in range(20)]
    iterList = [1]*len(queryId)
    docSinNameList = [i for j in list(docNameList.values()) for i in j]
    relevSinList = [i for j in list(relevList.values()) for i in j]

    #Exporting results to csv 'BRSout.csv' file
    df={'QueryId':queryId, 'Iteration':iterList, 'DocId':docSinNameList, 'Relevance':relevSinList}
    df = pd.DataFrame(df)
    df.to_csv('Output/QRels-BRS.csv',index=False)


#TF-IDF System
def tf_idf():

    #docNameList contains list of list of names of documents with relevance to queries
    docNameList = {}
    #relevList contains list of list of query relevance to documents
    relevList = {}
    #For loop to initialize docNameList and relevList
    for k,v in queryDict.items():
        docNameList[k]=[]
        relevList[k]=[]

    #noFiles contains the number of files
    noFiles = len(fileNames)
    #For loop to process all queries of the query file
    for query,qID in queryDict.items():
        #Remove HTML tags
        htmlTags = re.compile('<.*?>')
        queryText = re.sub(htmlTags, '', query)
        #Remove punctuation marks
        queryText = re.sub(r'[^\w\s]',' ',queryText)
        #Remove non-ASCII characters
        encodedQuery = queryText.encode("ascii","ignore")
        queryText=encodedQuery.decode()
        #Tokenization of the stream
        queryTokens = tokenizer(queryText)
        #Converting number tokens to words
        queryTokenNo=[]
        for j in queryTokens:
            if j.isdigit() and len(j)<4:
                j=n2w(int(j))
            queryTokenNo.append(j)

        queryTokens=queryTokenNo
        #Converting tokens into lower-case
        queryTokens = [token.lower() for token in queryTokens]
        #Stemming tokens
        queryTokens = [stemmer.stem(token) for token in queryTokens]
        #Removing stopwords from tokens except "and" and "or"
        queryTokens = [token for token in queryTokens if token not in StopWords]
        #Keeping tokens that are present in the files. Removing all others
        queryTokens = [token for token in queryTokens if token in uniqueTokens]
        tokens=queryTokens

        #queryTf contains the tf-idf score for tokens in the query
        queryTf = []
        #queryNorm is used to normalize the tf-idf score
        queryNorm = 0
        #For loop to get the queryTf
        for i in range(len(tokens)):
            tfIdf = (tokens.count(tokens[i])* math.log(noFiles/docFreq[tokens[i]]))
            queryTf.append(tfIdf)
            queryNorm=queryNorm + tfIdf**2
        queryNorm=math.sqrt(queryNorm)

        #For loop to normalize the queryTf
        for i in range(len(tokens)):
            queryTf[i] = queryTf[i]/queryNorm

        #tokenTfIdf contains the tokens tf-idf score in files
        tokenTfIdf = {}
        #For loop to get the tokentfIdf
        for i in range(len(fileNames)):
            docWordsTf=[]
            #For loop to get tf-idf for each file
            for token in tokens:
                tfIdf = (fileWords[i].count(token)*math.log(noFiles/docFreq[token]))
                docWordsTf.append(tfIdf)
            #For loop to normalize tf-idf
            for j in range(len(tokens)):
                docWordsTf[j] = docWordsTf[j]/invDocFreq[i]
            dp = 0
            #For loop to get dot product of tf-idf score for tokens in files and tf-idf score for tokens in the query
            for dW,qT in zip(docWordsTf, queryTf):
                dp = dp + dW*qT
            #Storing dot product in tokenTfIdf
            tokenTfIdf[i] = dp

        #Sorting dictionary as items according to value in descending order
        tokenTfIdf= dict(sorted(tokenTfIdf.items(), key=ig(1), reverse=True))
        
        #For loop to get the top documents with relevance>0 for all queries and their relevance(0 or 1)
        for i in range(20):
            docNameList[query].append(fileNames[(list(tokenTfIdf.keys()))[i]].split(".")[0])
            if tokenTfIdf[(list(tokenTfIdf.keys()))[i]]==0:
                relevList[query].append(0)
            else:
                relevList[query].append(1)
                
    #Creating lists of query id, iteration, doc id, relevance to output in csv files
    queryId =  [ele for ele in list(queryDict.values()) for i in range(20)]
    iterList = [1]*len(queryId)
    docSinNameList = [i for j in list(docNameList.values()) for i in j]
    relevSinList = [i for j in list(relevList.values()) for i in j]

    #Exporting results to csv 'TF-IDFout.csv' file
    df={'QueryId':queryId, 'Iteration':iterList, 'DocId':docSinNameList, 'Relevance':relevSinList}
    df = pd.DataFrame(df)
    df.to_csv('Output/QRels-TF-IDF.csv',index=False)


#BM25 System
def bm25():

    #docNameList contains list of list of names of documents with relevance to queries
    docNameList = {}
    #relevList contains list of list of query relevance to documents
    relevList = {}
    #For loop to initialize docNameList and relevList
    for k,v in queryDict.items():
        docNameList[k]=[]
        relevList[k]=[]

        
    #Count of number of files
    noFiles = len(fileNames)
    totLen = 0
    #Calculating total length
    for i in docLenName.keys():
        totLen=totLen+docLenName[i]
    avgLen = totLen/noFiles

    #Ideal Values for parameters k and b
    k=1.2
    b=0.75

    #For loop to process all queries of the query file
    for query,qID in queryDict.items():
        #Remove HTML tags
        htmlTags = re.compile('<.*?>')
        queryText = re.sub(htmlTags, '', query)
        #Remove punctuation marks
        queryText = re.sub(r'[^\w\s]',' ',queryText)
        #Remove non-ASCII characters
        encodedQuery = queryText.encode("ascii","ignore")
        queryText=encodedQuery.decode()
        #Tokenization of the stream
        queryTokens = tokenizer(queryText)
        #Converting number tokens to words
        queryTokenNo=[]
        for j in queryTokens:
            if j.isdigit() and len(j)<4:
                j=n2w(int(j))
            queryTokenNo.append(j)

        queryTokens=queryTokenNo
        #Converting tokens into lower-case
        queryTokens = [token.lower() for token in queryTokens]
        #Stemming tokens
        queryTokens = [stemmer.stem(token) for token in queryTokens]
        #Removing stopwords from tokens except "and" and "or"
        queryTokens = [token for token in queryTokens if token not in StopWords]
        #Keeping tokens that are present in the files. Removing all others
        queryTokens = [token for token in queryTokens if token in uniqueTokens]
        tokens=queryTokens

        #IDF for storing idf for BM-25
        idf = [0]*len(tokens)
        #For loop to calculate idf for all tokens
        for i in range(len(tokens)):
            if tokens[i] in docFreq:
                idf[i] = math.log(((noFiles-docFreq[tokens[i]]+0.5)/(docFreq[tokens[i]]+0.5))+1)

        #tokenTfIdf contains the BM25 scores
        tokenTfIdf = {}
        #For loop to get BM25 scores
        for i in range(len(fileNames)):
            tokenTfIdf[i] = 0
            #For loop to get the token tf-idf
            for j in tokens:
                if j in termFreq:
                    if i in termFreq[j]:
                        tF = termFreq[j][i]
                        tokenTfIdf[i] = tokenTfIdf[i]+((idf[tokens.index(j)]*(k+1)*(tF))/(tF+k*(1-b+b*(docLenName[fileNames[i]]/avgLen))))

        #Sorting dictionary as items according to value in descending order
        tokenTfIdf= dict(sorted(tokenTfIdf.items(), key=ig(1), reverse=True))

        #For loop to get the top documents with relevance>0 for all queries and their relevance(0 or 1)
        for i in range(20):
            docNameList[query].append(fileNames[(list(tokenTfIdf.keys()))[i]].split(".")[0])
            if tokenTfIdf[(list(tokenTfIdf.keys()))[i]]==0:
                relevList[query].append(0)
            else:
                relevList[query].append(1)

    #Creating lists of query id, iteration, doc id, relevance to output in csv files
    queryId =  [ele for ele in list(queryDict.values()) for i in range(20)]
    iterList = [1]*len(queryId)
    docSinNameList = [i for j in list(docNameList.values()) for i in j]
    relevSinList = [i for j in list(relevList.values()) for i in j]

    #Exporting results to csv 'BM-25out.csv' file
    df={'QueryId':queryId, 'Iteration':iterList, 'DocId':docSinNameList, 'Relevance':relevSinList}
    df = pd.DataFrame(df)
    df.to_csv('Output/QRels-BM25.csv',index=False)

brs()
tf_idf()
bm25()