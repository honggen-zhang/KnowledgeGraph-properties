#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:04:41 2020

@author: user1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:15:05 2020

@author: user1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 21:20:08 2020

@author: user1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:33:40 2020

@author: user1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:52:33 2020

@author: user1
"""
import os
import re
from collections import Counter
import spacy
from wordcloud import WordCloud
import csv
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from fuzzywuzzy import fuzz
import statistics 
import math
import numpy as np
import scipy
from nltk.corpus import wordnet as wn
import gensim 
from gensim.models import Word2Vec
import random
from sklearn.decomposition import IncrementalPCA
import seaborn as sns
model_file = '/home/user1/Desktop/IE_js/model_js'
model1 = gensim.models.Word2Vec.load(model_file)
import spacy
sp = spacy.load('en_core_web_sm')
import collections
from scipy.stats import norm
#def get_DgreeMatrix(graph):
    
    
#def get_AdjacentMatrix():


def write_node(str,tmp,po,ge,allnode):
    
    
    with open(str, 'w', newline='') as csvfile:
        fieldnames = ['id','label','interval','compoment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        for node in po:
            #print(node)
            node_label = allnode[node]
            writer.writerow({'id': node,'label':node_label,'interval':tmp,'compoment':1})
        for node in ge:
            #print(node)
            node_label = allnode[node]
            writer.writerow({'id': node,'label':node_label,'interval':tmp,'compoment':2})

def write_edge(str,tmp,H):
    
    
    with open(str, 'w', newline='') as csvfile:
        fieldnames = ['source','target','label','interval']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        edges = []
        label = nx.get_edge_attributes(H,'label')
        for edge in list(H.edges()):
            relation_label = label[edge]
            edges.append([edge,relation_label])
        print(edges[0][1])
            
        for node in edges:
            #print(node)
            #node_label = allnode[node]
            writer.writerow({'source': node[0][0],'target':node[0][1],'label':node[1],'interval':tmp})            

    

        
    

def get_allnodes(str):
    all_nodes = {}
    content_test = open(str, "r").readlines()
    for i in range(1,len(content_test)):
        lines = content_test[i]
        a=lines.split(',')
        all_nodes[a[0]]=a[1][:-1]
    
    return all_nodes

    
def creat_Digraph(alltriplets,all_nodes):
    #print(len(rest))
    G=nx.MultiDiGraph()



    alledges = []

    #print(allnodes)
    all_nodes = set(all_nodes)
    all_nodes = list(all_nodes)
    #print(len(all_nodes))
    for node in all_nodes:
        #print(node)
        G.add_node(str(node))

    for edge in alltriplets:
        edge=edge.split('@')
        #print(edge[0])
        alledges.append([edge[0], edge[1]])
        #if edge[0] in all_nodes and edge[1] in all_nodes:
            
        G.add_edge(str(edge[0]), str(edge[1]),label=str(edge[2]))
    #print(G)
    #nx.set_edge_attributes(G, labels, "labels")
    #nx.draw_networkx_nodes(G,graph_pos, node_size=150)
    graph_pos=nx.spring_layout(G,k=True)

    #fig, ax = plt.subplots()
    #nx.draw_networkx(G, graph_pos, node_size=150)
    #plt.show()
    return G,graph_pos

def creat_graph(alltriplets,all_nodes):
    #print(len(rest))
    G=nx.Graph()



    alledges = []

    #print(allnodes)
    all_nodes = set(all_nodes)
    all_nodes = list(all_nodes)
    #print(len(all_nodes))
    for node in all_nodes:
        #print(node)
        G.add_node(str(node))

    for edge in alltriplets:
        edge=edge.split('@')
        #print(edge[0])
        alledges.append([edge[0], edge[1]])
        if edge[0] in all_nodes and edge[1] in all_nodes:
            
            G.add_edge(str(edge[0]), str(edge[1]),label=str(edge[2]))
    #print(G)
    #nx.set_edge_attributes(G, labels, "labels")
    #nx.draw_networkx_nodes(G,graph_pos, node_size=150)
    graph_pos=nx.spring_layout(G,k=True)

    #fig, ax = plt.subplots()
    #nx.draw_networkx(G, graph_pos, node_size=150)
    #plt.show()
    return G,graph_pos

def read_file(str):
    nodes=[]
    triplets = []
    content_test = open(str, "r").readlines()
    for i in range(1,len(content_test)):
        
        lines = content_test[i]
        #lines = lines.lower()
        a=lines.split(',')
        #print(a)
        #b = str(a[1:3])
        subj_id = a[0]
        obj_id = a[1]
        relation = a[2]
        nodes.append(subj_id)
        nodes.append(obj_id)
        if subj_id!=obj_id:
            triplets.append(subj_id+'@'+obj_id+'@'+relation)
            
    return triplets,nodes
def write_data(str,datas):
    
    with open(str, 'w', newline='') as csvfile:
        fieldnames = ['date','data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        for data in datas:
            writer.writerow({'date': data[0],'data':data[1]})

filelist = os.listdir(r'/home/user1/Desktop/IE_js/sim/data_folder/JS/weakly component/edge/')
filelist.sort()
#all_nodes = []
time=[]
fieder_all = []
pos = []
neg = []
obj = []
mean_l = []
mean_t1 = []
mean_t2 = []
mean_v = []
frequence = []
frequence1 = []
#triplets_with_label = []
triplets_positive_all =[]
nodes_pos_all =[]
#all_nodes = get_allnodes('/home/user1/Desktop/IE_js/sim/sub_space/encode_all.csv')
divide = []
min_ratio = []
max_ratio = []
diameter_list = []
triplet_num = []
path_list = []
path_list_digraph = []

for j in range(len(filelist)):
    tmpfn=str(filelist[j])
    print(tmpfn)
    time.append(tmpfn[:-4])
    
        
    triplets_with_label = []
    head_tail = []
    #triplets_objective, nodes_obj = read_file('/home/user1/Desktop/IE_js/sim/sub_space/object_graph/'+tmpfn)
    triplets_positive, nodes_pos = read_file('/home/user1/Desktop/IE_js/sim/data_folder/JS/weakly component/edge/'+tmpfn)
    #triplets_positive_all = set(triplets_positive + triplets_positive_all)
    triplets_positive_all = list(triplets_positive)
    triplet_num.append(len(triplets_positive_all))
    #nodes_pos_all = nodes_pos + nodes_pos_all
    
    #graph_obj = creat_graph(triplets_objective,nodes_obj)
    #print(graph_obj.edges())
    #graph_pos,pos = creat_graph(triplets_positive_all,nodes_pos)
    #print('graph_pos')
    Digraph_pos,pos_Di = creat_Digraph(triplets_positive_all,nodes_pos)
    #print(nx.is_directed_acyclic_graph(Digraph_pos))
    #cycle = nx.find_cycle(Digraph_pos)
    
    #c = sorted(nx.strongly_connected_components(Digraph_pos), key=len, reverse=True)
    #c = sorted(nx.connected_components(graph_pos), key=len, reverse=True)[0]
    #c1 = max(nx.strongly_connected_components(Digraph_pos), key=len)
    #if list(c1)==list(c):
        #print('a')
    #c1= []
    #for i in list(c):
        #c.append()
        
    
    count = 0
    pair = 0
    d_list = []
    allpath= nx.shortest_path(Digraph_pos)
    #average_path = nx.average_shortest_path_length(Digraph_pos)
    
    allnode = list(set(nodes_pos))
    for source in allnode:
        for target in allnode:
            try:
                path = allpath[source][target]
                count = count + len(path)-1
                pair = pair+1
                d_list.append(len(path))
            except:
                pass
                
    average_path = count/pair
    #print(average_path)
    path_list.append(average_path)
    diameter = max(d_list)
    diameter_list.append(diameter)
    
    
    #path_list.append(average_path)


    #H = Digraph_pos.subgraph(list(c))
    #diameter = nx.diameter(Digraph_pos)
    #diameter_list.append(diameter)
    #path = nx.average_shortest_path_length(graph_pos)
    #try:
        #cylcle = nx.find_cycle(Digraph_pos,orientation="original")
        #print(list(cylcle))
    #except:
        #pass
    
    #path = nx.average_shortest_path_length(Digraph_pos)
    #path_list.append(path)
    #path_list_digraph.append(path)
    
    '''
    degree = list(graph_pos.degree())
    degree_frequence = []
    for i in degree:
        degree_frequence.append(i[1])
    sns.distplot(degree_frequence, kde=False)
    
    
    nodes =list(graph_pos.nodes())
    
    path_frequence = []
    try:
    
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                
                try:            
                    path_len = nx.shortest_path_length(graph_pos,source=nodes[i], target=nodes[j])
                    path_frequence.append(path_len)
                except:
                    pass
                
        #plt.hist(path_frequence,density=True)
        sns.distplot(path_frequence,fit=norm, kde=False)
    except:
        pass
    '''
    #pos = nx.spring_layout(Digraph_pos,k=True)
    #nx.draw_networkx(Digraph_pos, pos, node_size=150)
    #plt.show()
    
    #cut_value, partition = nx.stoer_wagner(H)
    #print(cut_value)
    #if len(partition[0]) < len(partition[1]):
        #sub_node = partition[1]
    #else:
        #sub_node = partition[0]
    #H = graph_pos.subgraph(sub_node)
        
    #cut_value, partition = nx.stoer_wagner(H)
    #cut_value, partition = nx.minimum_cut(H,'1725','6395')
'''
xlabelsnew = []
for i in range(0,len(time),2):
   
    null = ' '
    xlabelsnew.append(time[i])
    if i<=len(time):
        xlabelsnew.append(null)
        
plt.plot(time,path_list)
plt.ylabel('DiGraph_average_path2')
plt.xticks(range(0,len(time)),xlabelsnew,rotation=90)   
#plt.savefig('/home/user1/Desktop/IE_js/sim/data_folder/JS/weakly component/pics/DiGraph_average_path.png',dpi = 300) 
'''