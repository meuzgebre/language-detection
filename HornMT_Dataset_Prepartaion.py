# -*- coding: utf-8 -*-
"""
HornMT_Dataset_Preparation
Created on Mon Dec 12 01:25:16 2021

@author: Meuz G
"""
# Import libs
import pandas as pd

# Load HornMT dataset
file_path = '/data/HornMT.xlsx'
HornMT = pd.read_excel(file_path)
#HornMT.head(1)

# Preprocess the dataframe
eng = pd.DataFrame(HornMT['eng'])
aaf = pd.DataFrame(HornMT['aaf'])
amh = pd.DataFrame(HornMT['amh'])
orm = pd.DataFrame(HornMT['orm'])
som = pd.DataFrame(HornMT['som'])
tir = pd.DataFrame(HornMT['tir'])

# New Col With Language Type

eng.insert(1, 'Language', 'English')
eng = pd.DataFrame({'Text': eng['eng'], 'Language' : eng['Language']})

aaf.insert(1, 'Language', 'Afar')
aaf = pd.DataFrame({'Text': aaf['aaf'], 'Language' : aaf['Language']})

amh.insert(1, 'Language', 'Amharic')
amh = pd.DataFrame({'Text': amh['amh'], 'Language' : amh['Language']})

orm.insert(1, 'Language', 'Oromigna')
orm = pd.DataFrame({'Text': orm['orm'], 'Language' : orm['Language']})

som.insert(1, 'Language', 'Somalia')
som = pd.DataFrame({'Text': som['som'], 'Language' : som['Language']})

tir.insert(1, 'Language', 'Tigirigna')
tir = pd.DataFrame({'Text': tir['tir'], 'Language' : tir['Language']})


# tir.head(1)

# Mergining DataFrames
HornMT_DS = pd.concat([eng, aaf, amh, orm, som, tir], axis=0)
#HornMT_DS.tail()

# Export DataFrame
HornMT_DS.to_excel('data/HornMT_Langugae_Detection.xlsx', encoding='utf-8')

# End Of Code
