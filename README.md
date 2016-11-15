The Simpsons Members:
* Cameron Carlin
* Bradley Kenstler
* Alice Zhao
* Melanie Palmer

Formal Project Proposal Document: https://docs.google.com/document/d/1eGdP5rDdMAKx6cPPW8bdsVZ2iBccsE8Ai5N8HrW5sMI/edit

Data Source (link): https://www.kaggle.com/wcukierski/the-simpsons-by-the-data

# Running To-Do List:
* remove the ones where speaking_line (6th column) == False
* replace character_id with the actual character names from the character csv?
* same with location
* use character as Y, use location, normalized text, and word count to create features (also maybe episode id?)
* manually create corpus of person specific terms or phrases?

# Everything below needs updating
Data Description (#vars, #rows, types, etc):
* 1460 rows for train
* 1460 rows for test
* 79 variables

Problem we would like to solve with this data/ Data manipulations:
We would like to be able to accurately predict the prices of homes based off a variety of characteristics, while also creating aggregated variables based on factors such as building type, year built or remodeled, and so on. We will also create levels of variables (low, medium, high, etc) to create categories for predictors.

Algorithms to attempt:
* KNN-Clustering (dividing prices into bins)
* Linear Regression (for actual values)
* Logistic Regression (for low/high)
