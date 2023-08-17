#Importing pathlib library to get the paths to the files
import pathlib
#Importing re library to get ASCII characters from file by using regex from this library
import re
#Importing word_tokenize from nltk.tokenize library to tokenize the file contents
from nltk.tokenize import word_tokenize as tokenizer
#Importing Stemmer from nltk.stem library for stemming the file contents
from nltk.stem import PorterStemmer
#Importing stopwords from nltk.corpus to get a list of stopwords to remove from file contents
from nltk.corpus import stopwords as sW
import nltk
nltk.download('stopwords')
nltk.download('punkt')
#Getting the stopWords list for English
StopWords = set(sW.words('english'))
#Creating PorterStemmer() object
stemmer=PorterStemmer()
#Importing num2words library to convert numbers into words
from num2words import num2words as n2w
#importing Counter from collections library to get counts 
from collections import Counter
#Importing math library to perform some mathematical operations
import math
#Importing pickle library to store created content in pickle files
import pickle as pkl


#pathList contains the paths inside the 'english-corpora' directory
pathList = list(pathlib.Path("english-corpora").iterdir())
#filePathList contains list of paths to files inside current directory
filePathList = []
#Getting paths that lead to files inside 'english-corpora'
for i in range(len(pathList)):
    if(pathList[i].is_file()):
        filePathList.append(pathList[i])
#Getting names of files in path
fileNames = [str(f).split('\\')[1] for f in filePathList]


#uniqueWords is dictionary containing unique tokens of all files as keys and 0 as value
uniqueWords={}
#docLenName contains file name as key and it's length as value
docLenName={}
#totLen contains total length of all files
totLen=0
#fileWords contains list of all files' tokens
fileWords=[]
#For loop to process each file in set
for i in range(len(fileNames)):
    #Open current file to read
    currFile = open(filePathList[i], "r", encoding="utf8")
    #Read current file
    fileText=currFile.read()
    #Don't process file if it has length less than 1
    if(len(fileText)<=1):
        continue
    
    #Remove HTML tags
    htmlTags = re.compile('<.*?>')
    fileText = re.sub(htmlTags, '', fileText)
    #Remove punctuation marks
    punc = re.compile('[^\w\s]')
    fileText = re.sub(punc,' ',fileText)
    #Remove non-ASCII characters
    encodedFileText = fileText.encode("ascii","ignore")
    fileText=encodedFileText.decode()
    #Tokenization of the stream
    fileTokens = tokenizer(fileText)

    #Converting number tokens to words
    fileTokenNo=[]
    for j in fileTokens:
        if j.isdigit():
            j=n2w(int(j))
        fileTokenNo.append(j)
    
    fileTokens=fileTokenNo

    #Change all tokens to lower-case
    fileTokens = [token.lower() for token in fileTokens]
    #Stemming all tokens
    fileTokens = [stemmer.stem(token) for token in fileTokens]
    #Getting length of the current file
    fileLen= len(fileTokens)
    #Inputting entry to docLenName dictionary
    docLenName[fileNames[i]]=fileLen
    #Updatting the total length of the document
    totLen=totLen+fileLen
    #Removing stopwords from token list
    fileTokens = [token for token in fileTokens if token not in StopWords]
    #Adding current file tokens to fileWords
    fileWords.append(fileTokens)
    #Updating unique words dictionary
    uniqueWords.update({token:0 for token in fileTokens})
    #Close current file
    currFile.close()


#Getting unique tokens from all files
uniqueTokens = uniqueWords.keys()
#termFreq holds the term frequency for tokens
termFreq={}
#docFreq holds the document frequency for documents
docFreq={}
#Total number of files
noFiles = len(fileNames)
#Initializing termFreq and docFreq
for i in uniqueTokens:
    termFreq[i]={}
    docFreq[i]=0
#For loop to loop for 
for i in range(noFiles):
    cntr=Counter(fileWords[i])
    for j in cntr.keys():
        docFreq[j] = docFreq[j] + 1
        termFreq[j][i] = cntr[j]


#invDocFreq dictionary holds the inverse document frequencies for all files
invDocFreq={}
#For loop to get the inverse document frequency
for i in range(noFiles):
    wordFreq=0
    cntr=Counter(fileWords[i])
    #For loop to get the normalized sum
    for j in cntr.keys():
        wordFreq=wordFreq+(cntr[j]*math.log(noFiles/docFreq[j]))**2
    #Getting the normalized inverse document frequency
    invDocFreq[i]=(math.sqrt(wordFreq))


#Storing document frequency in docFreq.pkl
fl=open("pklFiles/docFreq.pkl", "wb")
pkl.dump(docFreq, fl)
fl.close()

#Storing document length and names in docLenName.pkl
fl=open("pklFiles/docLenName.pkl", "wb")
pkl.dump(docLenName, fl)
fl.close()
#Storing processed tokens for each file in fileWords.pkl
fl=open("pklFiles/fileWords.pkl", "wb")
pkl.dump(fileWords, fl)
fl.close()

#Storing inverse document frequency in invDocFreq.pkl
fl=open("pklFiles/invDocFreq.pkl", "wb")
pkl.dump(invDocFreq, fl)
fl.close()

#Storing token frequency in termFreq.pkl
fl=open("pklFiles/termFreq.pkl", "wb")
pkl.dump(termFreq, fl)
fl.close()