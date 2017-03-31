The Simpsons Members:
* Cameron Carlin
* Bradley Kenstler
* Alice Zhao
* Melanie Palmer

Google Slides Presentation: https://docs.google.com/presentation/d/16pyf2bN2KBMHmWfQ47IwyxvP_73TRdx7vIDEKpzpCwU/edit?usp=sharing

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
