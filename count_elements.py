# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:54:13 2019

@author: Kaai
"""
import pandas as pd
import numpy as np
import composition

df = pd.read_clipboard()
df.columns = ['formula']

index=['Al', 'B', 'C', 'Co', 'Cr', 'Fe', 'Hf', 'Ir', 'Mo', 'N', 'Nb', 'Ni', 'O', 'Os', 'P', 'Pt', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Ta', 'Tc', 'Th', 'Ti', 'V', 'W', 'Zr']

count_dict = {}

for element in index:
    count_dict[element] = 0

series = pd.Series(np.zeros(len(index)), index=index)
for formula in df['formula']:
    formula_dict = composition._element_composition(formula)
    for key in formula_dict.keys():
        count_dict[key] = formula_dict[key]

