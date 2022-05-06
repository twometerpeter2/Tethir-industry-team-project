import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import glob as glob
import warnings
warnings.filterwarnings("ignore")

df_ratios = pd.read_csv("C:/Users/Peter's 2nd PC/Downloads/ratio_data.csv")
df_ratios["background"] = df_ratios["background"].astype("string")

class test_generator: # library of tests for specific background determination
    
    def __init__(self, data):
        
        self._data = data
        
    def flame_test(self): # sets up test for determining if a flame is present or not
        
        df_flame = self._data
        df_flame["background"] = np.where(df_flame["background"].str.contains("flame"), "flame", "no flame")
        
        return df_flame
        
    def day_night_test(self): # sets up test for determining if it is day or night
        
        df_night = self._data
        df_night = df_night[df_night["background"].isin(["clear_test", "darkroom"])]
        df_night.reset_index(inplace = True, drop = True)
        
        return df_night
        
    def weather_test(self): # sets up test for determining the weather conditions
        
        df_weather = self._data
        df_weather = df_weather[df_weather["background"].isin(["clear_test", "light_cloud"])]
        df_weather.reset_index(inplace = True, drop = True)
        
        return df_weather
        
    def human_motion_test(self): # sets up test for determining if humans are present and in motion
        
        df_motion = self._data
        df_motion = df_motion[df_motion["background"].isin(["darkroom", "darkroom_ppl", "darkroom_ppl_moving"])]
        df_motion.reset_index(inplace = True, drop = True)
        
        return df_motion
    
class background_detection:
    
    def __init__(self, human_ratio, sunlight_ratio, ratio_data):
        
        self._human_ratio = human_ratio
        self._sunlight_ratio = sunlight_ratio
        self._ratio_data = ratio_data
        
    def nearest_neighbour(self):
        
        df_test = pd.DataFrame({"human ratio": [self._human_ratio], "sunlight ratio": [self._sunlight_ratio]})
        df_ref = pd.DataFrame({"human ratio": self._ratio_data["human ratio"], "sunlight ratio": self._ratio_data["sunlight ratio"]})
        
        nbrs = NearestNeighbors(n_neighbors = 5, algorithm = 'auto').fit(df_ref) # sets up 2D space for NN algorithm
        distances, indices = nbrs.kneighbors(df_test) # deploys NN algorithm for test data

        indices = np.array(indices).T.flatten()
        distances = np.array(distances).T.flatten() # turns NN outputs into useable arrays

        df_near = df_ratios.iloc[indices] # filters closest arrays (determined from NN) out of main ratios dataframe
        df_near.reset_index(inplace = True, drop = True)
        df_near["distance"] = distances # adds their distances from test data to the new dataframe
        df_near["weight"] = 1/distances # calculates the "weight" - used to determine background
        
        return df_near
        
    def conditions(self):
        
        df_near = background_detection.nearest_neighbour(self)
        
        df_near = df_near.groupby("background")["weight"].sum() # groups NN dataframe from above by background and sums the weights
        df_near = pd.DataFrame(df_near)
        df_near.reset_index(inplace = True)
        df_near = df_near.sort_values(by = "weight", ascending = False).reset_index(drop = True) # sorts dataframe by weight in descending order
        
        max_val = df_near.loc[0, "weight"] # prints largest weight
        
        arr_within = np.array(df_near[df_near["weight"] > 0.8 * max_val]) # used in case where two weights are similar, finds all backgrounds with weights larger than 80% of highest weight
        
        if len(arr_within) > 1: # if two backgrounds have similar weights...
            
            return "Two backgrounds weights too close to decide"
        
        elif all(x < 1 for x in df_near["weight"]): # if all weights are too small...
            
            return "No weights large enough to decide - test data too far away"
        
        else:
            
            return "Background is most likely: " + arr_within[0,0]

test = test_generator(data = df_ratios)

human_ratio = 1.75
sunlight_ratio = 2

a = background_detection(human_ratio, sunlight_ratio, df_ratios)
print(a.conditions())