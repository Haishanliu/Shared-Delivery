import networkx as nx
import osmnx as ox
import  xml.dom.minidom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# use NetworkX to prepare Graph(directed, each link has two weight: length and time)
#--- input: nodes(id, coordinate), edges(from,to,length,traveltime)


os.chdir(r"C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\network")
# load nodes
node = pd.read_csv("node_convert_to_degree.csv")
G = nx.DiGraph()
for i in range(len(node)):
    G.add_node(node.loc[i,'id'], size=0.01,weight = 0,coordinate =(node.loc[i,'Lat'],node.loc[i,'Lon']))
node_num = G.number_of_nodes()
#print('node_number=',node_num)

#lode links
link = pd.read_csv('Riverside_link.csv')

for j in range(len(link)):
    G.add_edge(link.loc[j,'X_from'],link.loc[j,'X_to'],id = link.loc[j,'X_id'],length = link.loc[j,'X_length'])
link_num = G.number_of_edges()
#print('link_number=',link_num)


path = nx.shortest_path(G,source=2431,target=2428,weight='length') # test_shortest_path
print('shortest path length:',path)
pathGraph = nx.path_graph(path)
length = 0
for ed in pathGraph.edges():
    print(ed,G.edges[ed[0],ed[1]]['length']) # loop over the edges in the shortes path, and record the attributes of each edges.
    length = length + G.edges[ed[0],ed[1]]['length']
print('shortest path length:',length )
print('total number of edges:',len(path))



# To do list for 11.12-11.19 Week 8
##  1. ready to do the shortest path. But some links only walk or bike? need to screen them or not? It is ok to keep them. Not a big portion.
##  2. Get familiar with NwtworkX -- read the documentation
##  3. add node information in the routing(for the customer and food provider) -- wait for the scenario setting, will do it later
##  4. find the way to record the shortest path edge-- done by loop over edges.
##  5. set ALNS with the Graph information
