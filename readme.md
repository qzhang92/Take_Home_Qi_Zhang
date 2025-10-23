# Introduction
This is the problem description and thoughts
# Content
### Background
We have a link on a web page. There might be another link after you click that link. Then you might have a button to download a file. We assume that we might have downloaded 100 files. Let's call them file1 to file100  
We have a program, A.exe. It is taking the 100 files as an input. Then generate 100 corresponding files, called 1.txt, 2.txt, 3.txt, ... 100.txt.
### Problem
We would like to group the output 100 files. By putting them into groups, we can better analyze them. We might be able to access the grouping rules and a sample dataset to group them based on the sample group rules.   
Challenge: Although sample dataset can help us with grouping, we do not have constant number of groups. For example, there might be 5 groups for the sample dataset, while the actuall 100 txt files actually need to be grouped into 10 groups. And we might have 10 groups today for the 100 files, but when 101.txt comes into the dataset, it does not belong to any of the 10 groups and we need to create group 11 for 101.txt.   
Use AI model and python to solve the problem
# Solution
### Assumption
We assume the file download and excution at A.exe is already done/considered to be out of scope
### Process
1.  **Read the .txt file**
2.  **Extract dataset**
    *   Listed some possible solutions of model and made comparation
    *   I selected the ```Word2Vec/FastText``` approach, followed by ```Averaging Word Vectors``` to create the final document embedding
3.  **Model Training & Initial Clustering**
    * DBSCAN Application
    * Cluster Center Calculation
4.  **New File Classification & Incremental Model Update**
    * New File Encoding
    * Distance and Threshold Check
    * Decision


## 
By Qi Zhang