#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:54:08 2021

@author: joshualambert
"""

import os
import itertools
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from visual import setting as se


se.set_rc_params()


df = pd.read_csv('avgs.csv')

df.topics.unique()

hilo = ['Common Property', 'Land Resources', 'Resources',
       'Local Management', 'Land and Livestock Commons',
        'Forests',
       'Software', 'Species', 'Plants', 'Urban']

changes = ['Common Property', 'Land Resources', 'Resources',
       'Local Management', 'Land and Livestock Commons', 'Groundwater Economics',
       'Farming', 'Games', 'Households', 'Conservation']





fig, ax = plt.subplots(figsize=(12,7))
d = df.loc[df.topics.isin(hilo)]
for t in d.topics.unique():
    
    d.loc[d.topics==t].drop(columns=['Unnamed: 0', ]).set_index('topics').T.plot(ax=ax, label=t)
                            
                            
    plt.xticks(rotation=45)
    plt.ylim(0, .14)
    
    #plt.title("Topic " + str(t), pad = 20)
    plt.ylabel("Topic Marginal Distribution")
    plt.legend(loc='upper center', ncol=5)
    fig.savefig("hilo.png")




fig, ax = plt.subplots(figsize=(12,7))
d = df.loc[df.topics.isin(changes)]
for t in d.topics.unique():
    
    d.loc[d.topics==t].drop(columns=['Unnamed: 0', ]).set_index('topics').T.plot(ax=ax, label=t)
                            
                            
    plt.xticks(rotation=45)
    plt.ylim(0, .14)
    
    #plt.title("Topic " + str(t), pad = 20)
    plt.ylabel("Topic Marginal Distribution")
    plt.legend(loc='upper center', ncol=5)
    fig.savefig("changes.png")
    