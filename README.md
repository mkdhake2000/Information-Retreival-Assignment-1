# **Assignment Submission by Mandar Dhake (21111405-mkdhake21@iitk.ac.in)**

This assignment contains 2 zips and 1 Makefile.

To run the assignment, use the command (first run the 'preprocessing-code.py' to create files required to run the assignment)

```bash
>> make run filename=Path
```


where `Path` is the location of the Query.txt.

```bash
>> make run
```


If it is not specified, then it will take the default Queries.txt, i.e., My own query list.

## **Dependencies:**
This assignment needs the following packages or libraries to be installed:
1. pandas
2. numpy
3. nltk
4. num2words
5. nltk.download('stopwords')
6. nltk.download('punkt')
7. PorterStemmer
8. math
9. re
10. operator
11. sys
12. pickle

To get the output, I have created an Output Folder where we will see 3 different Files.
1) Boolean Retrieval System (BRS.csv)
2) TF-IDF (TF-IDF.csv)
3) BM25 (BM25.csv)

We can take these files to evaluate our model.

## **Details about Zips:**
1) `21111047-ir-systems.zip`
2) `21111047-qrel.zip`

The first zip is our system file where we have implemented our system. It Contains 2 folders:
1) Code - Where I kept all the code.
2) Generated - Where I have kept all pickle files that are generated and used by our model.

Also, this contains 1 python file named run.py. This is the main file that calls all files and produces output.

The second zip contains my Queriestxt and QRels.csv, which are my ground truth values.

All the codes are documented well for better understanding.
