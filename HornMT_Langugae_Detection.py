# -*- coding: utf-8 -*-
"""
HornMT_Langugae_Detection.py
Created on Mon Dec 12 01:25:16 2021

@author: Meuz G
"""

# Import libs
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

# Loading the dataset
file = 'data/HornMT_Langugae_Detection.xlsx'
data = pd.read_excel(file)

# Value count for each language
data["Language"].value_counts()

# Separating the independent and dependant features
X = data["Text"]
y = data["Language"]
data.head(1)

# converting categorical variables to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# creating a list for appending the preprocessed text
data_list = []
# iterating through all the text
for text in X:
    # removing the symbols and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    # converting the text to lower case
    text = text.lower()
    # appending to data_list
    data_list.append(text)
    
# creating bag of words using countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()


# Train Test Splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# Model Creation and Prediction
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)


# Save The Model
'''
import pickle
filename = 'data/detection_model.sav'
pickle.dump(model, open(filename, 'wb'))
'''

# prediction 
y_pred = model.predict(x_test)

# model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print (f"Accuracy : {ac} \n Confusion Matrix : {cm}")


# ploting the confusion matrix
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()


# function for predicting language
def predict(text):    
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The langauge is in",lang[0])
    
# English
predict("Analytics Vidhya provides a community based knowledge portal for Analytics and Data Science professionals")

# Amharic
predict("የእስራኤል አየር ሃይል የተቃጣበትን የሚሳኤል እና የሞርታር ጥቃት ማላሽ ነው ሲል በጋዛ ሰርጥ ላይ ጥቃት መፈፀም ጀመረ።")

# Afar
predict("Israail qaran Cayli Misaailit Maysataka kee Morter Girah Gaazak Illa Israail Culusa Gacooral Maysataka Qimbise.")

# Oromigna
predict("Humni qilleensaa Isiraa'el haleellaa misaayelaa fi ibbiddaa ciccimaa irra ga'eef deebii kennuuf holqoota gara gaazaa geessan irratti haleellaa ni bana.")

# Somalia
predict("Ciidamada Cirka ee Israa’iil ayaa weerar ku qaaday marinnada dhulka hoostiisa ee Gaza, iyadoo ka jawaabaysa weerarrada gantaalaha iyo hoobiyayaasha lala beegsaday Israel.")

# Tigrigna
predict("ኣብ ኣፍጋኒስታን ቀንዲ መራሒ ኣልቃይዳን ካብ ኦሳማ ቢን ላደን ቀፂሉ ሳልሳይ ላዕለዋይ ኣዛዚ ኣቡ ኣል ያዚድ ከምዝቐተለ ተሓቢሩ።")