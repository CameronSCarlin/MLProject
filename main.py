###############################
###### Final Executable #######
###############################

import Validations, LogReg, KNNClass, RForest, SuppVec, Features
import pandas as pd
from Features import extract_features, top_words
import numpy as np


file1 = "simpsons_characters.csv"
file2 = "simpsons_episodes.csv"
file3 = "simpsons_locations.csv"
file4 = "simpsons_script_lines.csv"


characters = pd.read_csv(file1)
episodes = pd.read_csv(file2)
locations = pd.read_csv(file3)
lines = pd.read_csv(file4, low_memory=False, error_bad_lines=False, encoding='utf-8', warn_bad_lines=False)
lines = lines.iloc[ : , :13]


#subset of data we need
data = lines[lines.speaking_line=='true'][['location_id', 'normalized_text','character_id']].dropna()
#convert location_id from float to integer to string
data['location_id'] = [str(int(i)) for i in data['location_id']]

targets = data['character_id']

### Method to construct list of top words for feature construction is called below
### saved word list in words.csv and am loading to save time while testing

#words = np.array(top_words(data))
words = np.loadtxt('words.csv', delimiter=',', dtype='S')


#WILL RUN OUT OF MEMORY if ran on whole set, try on subset
#we should explore sparse matrices?

features = extract_features(data)

# select features
# features = Features.feature_selection(features,targets)


#simply test for Logisticregression
# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()
# model.fit(features, targets)
# print model.score

