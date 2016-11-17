The Simpsons Members:
* Cameron Carlin
* Bradley Kenstler
* Alice Zhao
* Melanie Palmer

Formal Project Proposal Document: https://docs.google.com/document/d/1eGdP5rDdMAKx6cPPW8bdsVZ2iBccsE8Ai5N8HrW5sMI/edit

Google Slides Presentation: https://docs.google.com/presentation/d/16pyf2bN2KBMHmWfQ47IwyxvP_73TRdx7vIDEKpzpCwU/edit?usp=sharing

### Running To-Do List:
* remove the ones where speaking_line (6th column) == False
* replace character_id with the actual character names from the character csv?
* same with location
* use character as Y, use location, normalized text, and word count to create features (also maybe episode id?)
* manually create corpus of person specific terms or phrases?

## About the Data:
*Data Source:* https://www.kaggle.com/wcukierski/the-simpsons-by-the-data

*Data Description:* This dataset contains the characters, locations, episode details, and script lines for approximately 600 Simpsons episodes, dating back to 1989. Contains 158,272 individual lines of speech.


### Problem we would like to solve with this data:
Using the lines of speech from every Simpsons episode ever, we aim to accurately classify which character is speaking based off what is spoken, where it is spoken, and so on.


### Methodology to solve this problem (in addition to existing features):
* Narrow down important locations, as particular characters tend to speak and hang out in particular areas.
* Use n-gram counts instead of just word counts. (Instead of “do” “n’t” “have” “a” “cow” “man”, “cow man” might more accurately predict who is speaking.


### Algorithms to attempt:
* Logistic Regression 
* KNN-Classification 
* Random forest
* SVM


## Criteria to determine the best model:
* Tuning accuracy for ‘main’ characters versus others: there are over 6,000 characters in the full collection of lines, where it would be ideal to tune the model to the more common characters for both accuracy and interest. 
* Misclassification rate, balance between true positive and false positive (ROC curves)
