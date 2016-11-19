###############################
###### Final Executable #######
###############################

import Validations, LogReg, KNNClass, RForest, SuppVec, Features
import pandas as pd

file1 = "simpsons_characters.csv"
file2 = "simpsons_episodes.csv"
file3 = "simpsons_locations.csv"
file4 = "simpsons_script_lines.csv"


characters = pd.read_csv(file1)
episodes = pd.read_csv(file2)
locations = pd.read_csv(file3)
lines = pd.read_csv(file4, low_memory=False, error_bad_lines=False)
lines = lines.iloc[ : , :13]

#subset of data we need
data = lines[lines.speaking_line=='true'][['location_id', 'normalized_text','character_id']].dropna()

target = data['character_id']

features = extract_feature(data)
