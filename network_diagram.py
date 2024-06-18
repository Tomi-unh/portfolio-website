#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 19:26:13 2023

@author: tadewuyi
"""
import networkx as nx
import numpy as np
import os
import pandas as pd
import datetime as dt
import glob
import pickle 
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_corr_matrix(correlation_result: list, datetime: str, number_of_stations: int = 494, steps: int = 5) -> dict:
    '''
    Get the correlation matrices from the results of the CCA and store these matrices into a dictionary.
    The key for the matrices are the timestep for each individual matrix. E.g, time t, t+1, t+2...
    Where time t is the beginning of the CCA analysis. 
    
    Parameters:
        ----------
        correlation_result: This is a list of lists containing the correlation coefficients between two stations.
        The list should be a square number and each sub-list in the list should represent the correlation coefficients
        between two stations.
        datetime: Date and time of the event. This is of the format 'yyyymmdd-hhmm', where the hhmm is the start
        hour and minute of the CCA analysis.
        number_of_stations: The number of stations being used for the CCA analysis. It's set to a default of 494.
        steps: The step between the first CCA analysis and the next one. Defaults to 1 for a normal rolling window.
        
        
        Returns:
        -------
        correlation_matrix: This is a dictionary of correlation matrices. Each one of these matrices corresponds 
        to a specific time step, with the appropriate label used as the key.
    '''
    
    datetime = datetime # Replace with your date string
    datetime_object = dt.datetime.strptime(datetime, '%Y%m%d-%H%M')
    
    n = number_of_stations
    
    
    #Get the length of the timesteps for the correlation matrix. 
    #This is done by taking the length longest sublist in the main list.
    length_of_result = len(max(correlation_result)) 
    
    corr_matrix = {}
    
    '''
    Make sure all the sub-lists in  the main list is all of the same length. Fill the empty ones with zeros.
    '''
    for ls in correlation_result:
        diff = length_of_result - len(ls)
        if len(ls) < length_of_result:
            ls.extend([0]*diff)
    
    list_array = np.array(correlation_result) #convert the list to array of 
    list_array = list_array.T
    
    for time in tqdm(range(length_of_result), total = length_of_result, desc = 'Prepping Correlation Matrix...'):
        ss = list_array[time].reshape(n,n)
        
        time_key = datetime_object + dt.timedelta(minutes = (steps)*time)
        corr_matrix[time_key] = ss
        
        
    return corr_matrix
            


def network(corr_matrix: np.ndarray, Time: dt.datetime, latitude: list, longitude: list, weight: float = 1, threshold: int = 0.99, path: str = None):
    """
    This function creates plot of networks between two stations, with a threshold setting of 0.9. Correlation
    values less than this will be ignored.

    Parameters:
      ----------
      corr_matrix (np.ndarray): The correlation matrix.
      latitude: This is a list of the latitude of the stations used in order that they were used.
      longitude: List of longitude for  the stations used in the analysis in a sorted format.
      threshold (float): The correlation threshold for creating edges between nodes (default: 0.7).
      weight (float): The constant weight value to assign to all edges (default: 1.0).
      Time (dt.datetime): The timestamp associated with the network graph.
      path (str): The path to save the generated plot. This is set to auto save to the parent directory and under
      the Network_Graph folder. If one doesn't exists, it is created. If the defined path given as an input doesn't
      exists, one is also created. 
  
      Returns:
      -------
          
      None
    """
    time_str = Time.strftime('%Y-%m-%d %H:%M:%S')
    
    yr = str(Time.year)
    
    
    if path is None:  
      path = f'../Network_Graphs/{yr}/'
      os.makedirs(path)
   
    
    num_of_nodes = corr_matrix.shape[1] #Define the number of nodes of possible connection.
    
    Graph = nx.Graph()
    
    Graph.add_nodes_from(range(num_of_nodes))

    # Add edges based on correlation values above the threshold
    for i in range(num_of_nodes):
        for j in range(i + 1, num_of_nodes):
            correlation = corr_matrix[i, j]
            if abs(correlation) >= threshold:
                Graph.add_edge(i, j, weight = weight)

    fig, ax = plt.subplots()
    
    pos = {i: (longitude[i], latitude[i]) for i in range(num_of_nodes)}
    nx.draw(Graph, pos, with_labels=False, node_size=20, edge_color = 'blue', ax = ax)
    plt.axis('on')
    plt.title(f"CCA Network Diagram. Threshold: {threshold}, Time: {time_str}")
    ax.set_xlabel('Latitude',fontsize = 15, fontweight = 'bold')
    ax.set_ylabel('Longitude', fontsize = 15, fontweight = 'bold')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    filename = f"CCA_Network_Graph_{time_str}.png"  
    filepath = os.path.join(path,filename)
    plt.savefig(filepath)
    plt.close(fig)
    
    

def get_latlon(path: str, pattern: str = '/*.feather') -> list:
  """
  Get the latitude and longitude of each station and return a list of these values. 
  The latitdue and longitude values are sorted alphabetically for the station names. 
  
  Parameter:
    ----------
    path: Path to data folder.
    pattern: data file extension. Default is .feather
    
    Returns:
      --------
      Latitude, Longitude: 2 seperate list of latitude and longitude for the stations. 
  """
  lat = []
  lon = []
  
  station_list = []
  for file in glob.glob(f'{path }/{pattern}'):
    station_list.append(file)
  
  
  station_list.sort()
  
  for file in station_list:
    df = pd.read_feather(file)
    
    
    '''
    Get the first non NAN values in the dataframe GEOLAT and GEOLON columns. This prevents putting nan values
    in the output. The Values should all be the same so it doesn't matter the specific value stored in the return
    as long as it isn't nan
    '''
    lat_index = df['GEOLAT'].first_valid_index()
    lon_index = df['GEOLON'].first_valid_index()
    
    
    
    lat.append(df.GEOLAT[lat_index])
    lon.append(df.GEOLON[lon_index])
    
  return lat, lon


    
    
def main(path: str, datetime: str, lat: list = None,lon: list = None, file_path: str = None, number_of_station: int = 494):
  """
  Main function for this file. Takes in the data and it's corresponding start datetime, along with the path
  and creates graphs of the network between the stations for  the specific event. The data that contains a list of correlation matrix is 
  converted into a dictionary where the key is the start time for the respective 128 minute window. The resulting dictionary is returned 
  as an output of the function. This is then looped through and passed into the network function to create the network diagram. 
  
  Parameter:
    ----------
    data: This is a list of lists. Each sub-list is the correlation for two stations for each running 
    window. 
    datetime: String of the starting time of the event in the format of 'yyyymmdd-hhmm'.
    number_of_station: Number of station for the analysis.
    path: Path to the data file. This should include the name of the file with the extension (.pickle) as well. 
  
  """
  yr = datetime[0:4]
  
  if file_path is None:  
    file_path = f'../Network_Graphs/{yr}/'
    
  if not os.path.exists(file_path):
    os.makedirs(file_path)
   
 
  with open(path, 'rb') as pickle_file:
    data = pickle.load(pickle_file) 
    
  if lat is None or lon is None:
    lat,lon = get_latlon('../data/')
      
  corr_matrix_dic = extract_corr_matrix(data,datetime)
  
  for key, value in tqdm(corr_matrix_dic.items(), desc ='Creating network graphs'):
    network(value,key, lat, lon, path = file_path)


if __name__ == '__main__':
  print('This script is being run as the main program...')
  main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    