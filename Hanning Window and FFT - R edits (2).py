import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
#import scipy.fft as fft
from sklearn.neighbors import NearestNeighbors
import glob as glob
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "Times New Roman"})

class detection_algorithm:
    
    def __init__(self, file, data_set, savgol_window_length, savgol_polyorder, sample_freq, num_bin):
        
        self._dataframe = pd.read_csv(file, skiprows = 34) # converts file to dataframe
        self._data_set = str(data_set) # sets the dataset used (which sensor we're looking at)
        self._window_length = savgol_window_length # length of savgol windows that smoothing function uses
        self._polyorder = savgol_polyorder # polynomial order for smoothing function
        self._max_dect_freq = sample_freq / 2
        self._num_bin = num_bin
        
    def sensor_selector(self):
        
        # sensor = 2
        
        if int(self._data_set) == 1: # connects dataframe column to which sensor is being analysed
            sensor = 3
            return str(sensor)
        elif int(self._data_set) == 2:
            sensor = 2
            return str(sensor)
        elif int(self._data_set) == 3:
            sensor = 1
            return str(sensor)
    
    def dataframe_prep(self):
        
        df = self._dataframe
        
        df = df.iloc[:-6] # cuts out useless bottom text (imported from csv)
        df[["time", "data1", "data2", "data3", "error1", "error2", "error3"]] = df.iloc[:, 0].str.split("\t", expand = True) # splits data into its constituent columns
        df = df.iloc[:, 1:] # gets rid of first column (raw important data)
        df.reset_index(inplace = True)
        
        df = df.astype("int64")
        df = df.iloc[350:] # removes first 350 rows to reduce sensor start-up errors

        df["time"] = df["time"] - df.iloc[0]["time"] # zeros the time on first row
        df["time"] = df["time"] / 1000 # converts ms to s
        df = df[df.index % 10 == 0]
        df["data_SAV"] = savgol_filter(df["data%s" % (self._data_set)], window_length = self._window_length, polyorder = self._polyorder, mode = "mirror") # applies smoothing window
        df.drop(columns = "index", inplace = True)
        
        return df
    
    def raw_data_plot(self):
        
        """
        This plots the smoothed data from above
        """
        
        df = detection_algorithm.dataframe_prep(self)
        sensor = detection_algorithm.sensor_selector(self)
        
        plt.plot(df["time"], df["data%s" % (self._data_set)])
        plt.plot(df["time"], df["data_SAV"])
        plt.xlabel("Time (s)")
        plt.tick_params(axis = "x", bottom = True, top = True, direction = "in")
        plt.ylabel("Voltage (mV)")
        plt.tick_params(axis = "y", left = True, right = True, direction = "in")
        plt.title("Raw data plot - Sensor %s" % (sensor))
        
    def smoothed_data_plot(self):
        
        """
        This plots the smoothed data from above
        """
        
        df = detection_algorithm.dataframe_prep(self)
        sensor = detection_algorithm.sensor_selector(self)
        
        plt.plot(df["time"], df["data_SAV"], color = "k")
        plt.xlabel("Time (s)")
        plt.tick_params(axis = "x", bottom = True, top = True, direction = "in")
        plt.ylabel("Voltage (mV)")
        plt.tick_params(axis = "y", left = True, right = True, direction = "in")
        plt.title("Smoothed data plot - Sensor %s" % (sensor))
    
    def hanning_window(self):
        
        df = detection_algorithm.dataframe_prep(self)
        
        hanning = np.hanning(len(df)) # creates table of hanning values to multiplied by our data
        df["hanning"] = hanning * df["data_SAV"] # applies the hanning values to our data
        
        return df
    
    def hanning_plot(self):
        
        """
        Plots the hanned data
        """
        df = detection_algorithm.hanning_window(self)
        sensor = detection_algorithm.sensor_selector(self)
        
        plt.plot(df["time"], df["hanning"], color = "k")
        plt.xlabel("Time (s)")
        plt.tick_params(axis = "x", bottom = True, top = True, direction = "in")
        plt.ylabel("Voltage (mV)")
        plt.tick_params(axis = "y", left = True, right = True, direction = "in")
        plt.title("Hanning window applied to data - Sensor %s" % (sensor))
    
    def fourier_transform(self):
        
        df = detection_algorithm.hanning_window(self)
        
        fourier = np.fft.rfft(np.array(df["hanning"])) # applies fast fourier transform of hanning window data
        df_fourier = pd.DataFrame({"fourier": fourier}) # puts fourier data into dataframe
        df_fourier["fourier_abs"] = np.abs(df_fourier["fourier"]) # takes absolute of fourier data to remove imaginary values
        
        N = len(df_fourier)
        n = np.arange(N) # creates a list of 1 to length of fourier data
        time_period = N/self._max_dect_freq # calculates the tiome period of each data point
        df_fourier["freq"] = (n/time_period) # calculates frequency of each data point
        
        df_fourier = df_fourier.iloc[100:] # removes top 100 rows of data to get rid of enourmous values 
        
        return df_fourier
    
    def fourier_plot(self):
        
        """
        Plots fourier data
        """
        
        df_fourier = detection_algorithm.fourier_transform(self)
        sensor = detection_algorithm.sensor_selector(self)
        
        plt.plot(df_fourier["freq"], df_fourier["fourier_abs"], color = "k")
        
        plt.xlabel("Frequency (Hz)")
        plt.tick_params(axis = "x", bottom = True, top = True, direction = "in")
        plt.ylabel("Intensity (V s)")
        plt.tick_params(axis = "y", left = True, right = True, direction = "in")
        plt.title("Fourier transform of Hanning window - Sensor %s" % (sensor))
    
    def binning_fourier(self):
        
        df_fourier = detection_algorithm.fourier_transform(self)
        
        bins = np.linspace(1, self._num_bin, self._num_bin) # creates the bins for the data to be put in
        
        df_fourier["hist"] = pd.cut(df_fourier.index, bins = self._num_bin, labels = bins) # identifies which bin each data row should be in

        df_fourier_bins = df_fourier.groupby("hist")["fourier_abs"].mean().reset_index() # puts the data into the bins (averages the data for each bin)
        df_fourier_bins["hist"] = df_fourier_bins["hist"].astype("str")

        n_bins = np.arange(self._num_bin) # creates a list of 1 to number of bins
        time_period_bins = self._num_bin/self._max_dect_freq # calculates the time period for each bin
        df_fourier_bins["freq"] = (n_bins/time_period_bins) # calculates the frequency for each bin
        
        return df_fourier_bins
    
    def fourier_bin_plot(self):
        
        """
        Plots the binned fourier data
        """
        
        df_fourier_bins = detection_algorithm.binning_fourier(self)
        df_fourier_bins["log_fourier_abs"] = np.log10(df_fourier_bins["fourier_abs"])
        sensor = detection_algorithm.sensor_selector(self)
        
        plt.bar(
            df_fourier_bins["freq"],
            df_fourier_bins["fourier_abs"],
            width = df_fourier_bins.iloc[1]["freq"] - df_fourier_bins.iloc[0]["freq"],
            color = "k"
            )
        plt.xlabel("Frequency (Hz)")
        plt.tick_params(axis = "x", bottom = True, top = True, direction = "in")
        plt.ylabel("Intensity (V s)")
        plt.tick_params(axis = "y", left = True, right = True, direction = "in")
        plt.title("Fourier transform of Hanning window - Sensor %s" % (sensor))
        plt.xlim(-0.5, 100)
        # plt.ylim(0, 0.001)
        
    # def nearest_neighbour(self):
        
        # df_test = pd.DataFrame({"x": [1], "y": [2]})
        # df_base = pd.DataFrame({"x" : [1,3,4,2,3,2], "y" : [4,3,1,3,2,1]})
        
        # nbrs = NearestNeighbors(n_neighbors = 2, algorithm = 'auto').fit(df_base)
        # distances, indices = nbrs.kneighbors(df_test)
        
        # return distances, indices
        
    def fourier_summation(self):
    
        df_fourier = detection_algorithm.binning_fourier(self)    
        
        if self._data_set == "1":
           summed =  df_fourier.loc[(df_fourier["freq"]>=6) & (df_fourier["freq"]<=16) , "fourier_abs"].sum() 
           # flame.append(self.summed)
           # Flame frequency ranges

        elif self._data_set == "2":
           summed =  df_fourier.loc[(df_fourier["freq"]>=4) & (df_fourier["freq"]<=6) , "fourier_abs"].sum() 
           # sunlight.append(self.summed)
           # Sunlight frequency range

        elif self._data_set == "3":
           summed =  df_fourier.loc[(df_fourier["freq"]>=0) & (df_fourier["freq"]<=2) , "fourier_abs"].sum() 
           # human.append(self.summed)
           # Human frequency range
           
        else:
            print("dataset out of range")
        
        return summed        

file_name = "C:/Users/Ruair/OneDrive/Documents/Industry project data dump/2022-04-26/2022-04-26-13-09-19_test_flame_3.csv"
sample_freq = 200
num_bin = 100

a = detection_algorithm(file = file_name, data_set = 3, savgol_window_length = 1551 , savgol_polyorder = 4, sample_freq = sample_freq, num_bin = num_bin)
# print(sum(a.binning_fourier()["fourier_abs"].head(11)))
# print(a.fourier_bin_plot())
# print(a.fourier_summation())

a1 = detection_algorithm(file = file_name, data_set = 1, savgol_window_length = 1551 , savgol_polyorder = 4, sample_freq = sample_freq, num_bin = num_bin)
a2 = detection_algorithm(file = file_name, data_set = 2, savgol_window_length = 1551 , savgol_polyorder = 4, sample_freq = sample_freq, num_bin = num_bin)
a3 = detection_algorithm(file = file_name, data_set = 3, savgol_window_length = 1551 , savgol_polyorder = 4, sample_freq = sample_freq, num_bin = num_bin)

sunlight_rejection_ratio = a1.fourier_summation()/a2.fourier_summation()
human_rejection_ratio = a1.fourier_summation()/a3.fourier_summation()

print(sunlight_rejection_ratio)
print(human_rejection_ratio)

# print(a.nearest_neighbour()[0])

folder = r"C:\\Users\Ruair\OneDrive\Documents\Industry project data dump\uncertainties"

files = glob.glob(folder + '/*.csv')

# flame = []
# human = []        
# sunlight = []
# filename =[]

fourier_summation_array_1 = [] # flame
fourier_summation_array_2 = [] # human
fourier_summation_array_3 = [] # sunlight


# loop through list of files and read each one into a dataframe and append to list
for path in files:
    
    file_name = "{}".format(path)
    sample_freq = 200
    num_bin = 100
      
    a1 = detection_algorithm(file = file_name, data_set = 1, savgol_window_length = 1551 , savgol_polyorder = 4, sample_freq = sample_freq, num_bin = num_bin)
    a2 = detection_algorithm(file = file_name, data_set = 2, savgol_window_length = 1551 , savgol_polyorder = 4, sample_freq = sample_freq, num_bin = num_bin)
    a3 = detection_algorithm(file = file_name, data_set = 3, savgol_window_length = 1551 , savgol_polyorder = 4, sample_freq = sample_freq, num_bin = num_bin)
# print(sum(a.binning_fourier()["fourier_abs"].head(11)))
    # print(a.fourier_bin_plot())
    # print(a.sensor_selector())
    fourier_summation_array_1.append(a1.fourier_summation())
    fourier_summation_array_2.append(a2.fourier_summation())
    fourier_summation_array_3.append(a3.fourier_summation())
    
df_fourier_summation = pd.DataFrame(data = {"sensor1": fourier_summation_array_3, "sensor2": fourier_summation_array_2, "sensor3": fourier_summation_array_1})

df_fourier_summation["human_rej_ratio"] = df_fourier_summation["sensor3"] / df_fourier_summation["sensor1"]
df_fourier_summation["sunlight_rej_ratio"] = df_fourier_summation["sensor3"] / df_fourier_summation["sensor2"]
   
# print(df_fourier_summation)