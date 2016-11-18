###############################
###### Final Executable #######
###############################

import Validations, LogReg, KNNClass, RForest, SuppVec, Features
import pandas as pd

file = "simpsons_characters.csv"
file2 = "simpsons_episodes.csv"
file3 = "simpsons_locations.csv"
file4 = "simpsons_script_lines.csv"


characters = pd.read_csv(file)
episodes = pd.read_csv(file2)
locations = pd.read_csv(file3)
lines = pd.read_csv(file4, low_memory=False)
