The Simpsons Members:
* Cameron Carlin
* Bradley Kenstler
* Alice Zhao
* Melanie Palmer

Project Outline: https://docs.google.com/document/d/13vzFY9aZb9fmVQO8pOMGa9x8-LePPZWbUKHuZlGuc4E/edit?usp=sharing

Formal Project Proposal Document: https://docs.google.com/document/d/1eGdP5rDdMAKx6cPPW8bdsVZ2iBccsE8Ai5N8HrW5sMI/edit

Google Slides Presentation: https://docs.google.com/presentation/d/16pyf2bN2KBMHmWfQ47IwyxvP_73TRdx7vIDEKpzpCwU/edit?usp=sharing

### To-Do List for presentation/outline:
* Explanation of TFIDF attempt/implementation (add in slide, Alice or Brad)
* Explanation of PorterStemmer attempt/implementation
* Explanation of frequency of locations used
* "            " attempt using 'most common' words spoken by each class

### To-Do List for adding into slides:
* add in Misclassification rates for 1vAll, 4vAll, 10vAll, and Everyone
* add in future attempts
* add in the issues we had with the data itself

## About the Data:
*Data Source:* https://www.kaggle.com/wcukierski/the-simpsons-by-the-data

*Data Description:* This dataset contains the characters, locations, episode details, and script lines for approximately 600 Simpsons episodes, dating back to 1989. Contains approximately 131,000 individual lines of speech.


### Problem we would like to solve with this data:
Using the lines of speech from every Simpsons episode ever, we aim to accurately classify which character is speaking based off what is spoken and where.


### Methodology to solve this problem:
* Narrow down important locations, as particular characters tend to speak and hang out in particular areas.
* Criteria for word frequency and document frequency (via TFIDF)
* Stemming words to turn down the count of 'unique' words


### Algorithms Attempted:
* Logistic Regression (L1 and L2)
* KNN-Classification 
* Random forest
* SVM
* Decision Tree
* AdaBoost

## Criteria to determine the best model:
* Tuning accuracy for ‘main’ characters versus others: there are over 6,000 characters in the full collection of lines, where it would be ideal to tune the model to the more common characters for both accuracy and interest. 
* Misclassification rate, balance between true positive and false positive
