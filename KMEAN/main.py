#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 01:55:37 2018
@author: ArijitSinha

Question 1 - During the third week, you will analyze the quality of the clustering. 
• Write a code to calculate the individual and total error rate of your 2 clusters. 
• Submit final report

Name - Arijit Sinha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Function - Details about the Null/ Blank Values
def contblk(x):
    print('Details about the Null/ Blank Values in the Dataset')
    print()
    print(np.sum(x.isna()))
    print()
    
    return(x)

#Function - Replace Null/ Blank Values            
def repbnk(x):
    npval = np.around(np.nanmean(x['Bare Nuclei'], dtype = float), 2)
    print('The NaN Value will be replaced with', npval)
    #npval = np.around((x['Bare Nuclei'].mean()), 2)
    x['Bare Nuclei'].replace(np.nan, npval, inplace=True)
    print()
    print('Validate if the NaN has been replaced with or still exist')
    print()
    
    print(x.loc[x['Bare Nuclei'].isna()])
    
    return (x)

#Function - Calculate the statistics 
def statdf(x):
    print()
    print('Statistics from the Dataset')
    print()
    std = np.std(x)
    mea = np.mean(x)
    var = np.var(x)
    med = (np.median(x, axis = 0))
    
    return (std, mea, var, med)

#Function - Plot Histogram 
def histplt(x):
    fig = plt.figure(figsize=(14, 32))

    ax1 = fig.add_subplot(9, 2, 1)
    ax2 = fig.add_subplot(9, 2, 2)
    ax3 = fig.add_subplot(9, 2, 3)
    ax4 = fig.add_subplot(9, 2, 4)
    ax5 = fig.add_subplot(9, 2, 5)
    ax6 = fig.add_subplot(9, 2, 6)
    ax7 = fig.add_subplot(9, 2, 7)
    ax8 = fig.add_subplot(9, 2, 8)
    ax9 = fig.add_subplot(9, 2, 9)
    
    ax1.hist(x["Clump Thickness"], bins=10, color = "b", alpha = 0.5)
    ax2.hist(x["Uniformity of Cell Size"], bins=9, color = "b", alpha = 0.5)
    ax3.hist(x["Uniformity of Cell Shape"], bins=10, color = "b", alpha = 0.5)
    ax4.hist(x["Marginal Adhesion"], bins=10, color = "b", alpha = 0.5)
    ax5.hist(x["Single Epithelial Cell Size"], bins=9, color = "b", alpha = 0.5)
    ax6.hist(x["Bare Nuclei"], bins= 8, color = "b", alpha = 0.5)
    ax7.hist(x["Bland Chromatin"], bins=10, color = "b", alpha = 0.5)
    ax8.hist(x["Normal Nucleoli"], bins=10, color = "b", alpha = 0.5)
    ax9.hist(x["Mitoses"],bins=5, color = "b", alpha = 0.5)

    ax1.set_ylabel("Frequency")
    ax2.set_ylabel("Frequency")
    ax3.set_ylabel("Frequency")
    ax4.set_ylabel("Frequency")
    ax5.set_ylabel("Frequency")
    ax6.set_ylabel("Frequency")
    ax7.set_ylabel("Frequency")
    ax8.set_ylabel("Frequency")
    ax9.set_ylabel("Frequency")


    ax1.set_xlabel("Clump Thickness")
    ax2.set_xlabel("Uniformity Cell Size")
    ax3.set_xlabel("Uniformity Cell Shape")
    ax4.set_xlabel("Marginal Adhesion")
    ax5.set_xlabel("Single Epithelial Cell Size")
    ax6.set_xlabel("Bare Nuclei")
    ax7.set_xlabel("Bland Chromatin")
    ax8.set_xlabel("Normal Nucleoli")
    ax9.set_xlabel("Mitoses")


    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=2.5)
    plt.show()

#Function - Plot Correlation Matrix
def corrmatrix(x):
    corr = x.corr()

    # plot the heatmap
    sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)

#Function - Select Random two datapoints as means
def picktwocor(x):
    k = 2
    c = x.sample(k)
    #d = [c.iloc[0].values, c.iloc[1].values]
    randomu1 = c.iloc[0].values
    randomu2 = c.iloc[1].values
    #print("Initial Random Mean selected are :")
    #print('randomu1 :', randomu1)
    #print('randomu2 :', randomu2)
    #plt.scatter(u1, u2, color = 'green')
    return (randomu1, randomu2)
    # Alternate code - 
    #cn = x.iloc[np.random.choice(np.arange(len(x)), k, False)]
    #dn = [cn.iloc[0].values, cn.iloc[1].values]
    #un1 = cn.iloc[0].values
    #un2 = cn.iloc[1].values
    #plt.scatter(un1, un2, color = 'red')

#Function - Assignment of Predicted Clusters
def assigndf(x, u1, u2):
    PredictedClass = []
    for i in range(len(x)):
        distu1 = np.around((np.sqrt(sum(((x.iloc[i].values)-(u1))**2))),2)
        distu2 = np.around((np.sqrt(sum(((x.iloc[i].values)-(u2))**2))),2)
        if distu1<distu2:
            PredictedClass.append(2)
        else:
            PredictedClass.append(4)    
    
    return(PredictedClass)

#Function - Updated mean from clusters
def updatemean(x, pc):

    x['PreCls'] = pd.Series(pc, index=x.index)
    dfwith2 = x[x['PreCls']==2]
    dfwith4 = x[x['PreCls']==4]
    
    #print(len(dfwith2), len(dfwith4))
    
    utwoar = np.mean(dfwith2[:9], axis = 1).values
    ufourar = np.mean(dfwith4[:9], axis = 1).values
    
    #print(len(utwoar), len(ufourar))
    x.drop('PreCls', axis = 1, inplace  = True)
    return(x, utwoar, ufourar)

#Function - Recalculation and Iteration for 1500 runs or compare previous with curent predicted values
def recalmeanfifty(x, col, pc):
    i = 0
    while (i <= 1499):
        x1, a, b = updatemean(x, pc)
        #print(len(a), len(b))
        newpc = assigndf(x1, a, b)
        if pc == newpc:
            return(newpc)
            break
        else:
            pc = newpc
            x1, a, b = updatemean(x1, newpc)
            newpc = assigndf(x1, a, b)
        
        i = i+1
        print()
        print("-----Final Means-------")
        print("mu2:", a)
        print("mu4:", b)    
    return(newpc)
#Function - ErrorRate
def ErrorRate(actclass,predclass):
    counttwo= 0
    countfour = 0
    Notmatch = 0
    cbenign = 0
    cmalignant = 0
    TotalDatapoint = len(actclass)
    for i in range(len(actclass)):
        if (actclass[i] ==2 and predclass[i]==4):
            counttwo +=1
        elif (actclass[i]==4 and predclass[i]==2):
            countfour +=1
            
    for i in range(len(actclass)):
        if (predclass[i]==2):
            cbenign +=1
        else:
            cmalignant +=1

    for i in range(len(actclass)):
        if (actclass[i]!=predclass[i]):
            Notmatch +=1

    print()       
    print("Not Matching Benign:", counttwo)
    print() 
    print("Not Matching Malignant", countfour)
    print() 
    print("Total number of Benign:", cbenign)
    print() 
    print("Total number of Malignant:", cmalignant)
    print() 
    print("Total numberof Records", TotalDatapoint)
    print() 
    print("Total Not Matched:", Notmatch)

    errorB = counttwo/cbenign

    errorM = countfour/cmalignant

    TotalErrorRate = Notmatch/TotalDatapoint
    print() 
    print("Error with Benign :", np.around(errorB,5))
    print() 
    print("Error with Malignant :",np.around(errorM, 5))
    print() 
    print("Total Error Rate :", np.around(TotalErrorRate,5))

def main():
    print('The Program for Implementing “k-means” algorithm for Wisconsin Breast Cancer data using Python.')
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/'
          'breast-cancer-wisconsin.data')
    df = pd.read_csv(url, 
                     header=None, names=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 
                                         'Uniformity of Cell Shape', 'Marginal Adhesion', 
                                         'Single Epithelial Cell Size', 'Bare Nuclei', 
                                         'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 
                                         'Class'], na_values="?")
    
    dfcol = df['Class'].copy()
    
    
    df = df.set_index('ID')
    
    m,n = df.shape
    print()
    print('The Rows of the dataset is', m)
    print('The number of Attributes in the dataset is', n)
    print()
    
    # Information on Dataset loaded
    print('Information on Dataset Loaded')
    print(df.info())
    
    print()
    dforiginal = df
    print('Keep the copy of Origina dataset', dforiginal.shape)
    print()

    # Data Cleaning activties
    # Count the details on Null and ? in the data columns
    df = contblk(df)
    print('Datasection has the null Values')
    print()
    print(df.loc[df['Bare Nuclei'].isna()])

    # replace Nan and ? in the Data column
    dfwb = repbnk(df)
    dfwb.drop('Class',axis=1, inplace = True)
    # Draw 9 Historgram
    
    histplt(dfwb)
    
    # Data Statistics Details 
    # Calcuate the Statistics from dataset columns
    
    stdf, meaf, varf, medf = statdf(dfwb)
    print('Standard Deviation of each of the attributes A2 to A10')
    print()
    print(stdf)
    print()
    print('Mean of each of the attributes A2 to A10')
    print()
    print(meaf)
    print()
    print('Variance of each of the attributes A2 to A10')
    print()
    print(varf)
    print()
    print('Median of each of the attributes A2 to A10')
    print()
    col_names = df.columns.tolist()    
    for i in range(0,9):
        print(col_names[i].ljust(33) + str(medf[i]).ljust(2))

    print()
    
    # Draw Correlational Matrix 
    #print("Correlational Matrix")
    #corrmatrix(dfwb)
    #print()
    
    picktwocor(dfwb)
    
    arr1, arr2 = picktwocor(dfwb)
    
    predclassint = assigndf(dfwb, arr1, arr2)

    #predclass = recalmeanfirst(dfwb, predclassint)
    predclassfifty = recalmeanfifty(dfwb, dfcol, predclassint)
    
    #Print first 20 values of ID, Actual Class and final Predicted values
    print()
    print("----------------Cluster Assignments------------")
    print("ID     ", "Class ", "Predicted Class")
    for i in range(0,21):
        print(dfwb.index[i],'  ', dfcol[i],'        ', predclassfifty[i])
        
    # Calculate the error rate between the actual Class and Predicted Class
    print()
    print("Calculate the Individual & Total Error Rate - Both Clusters")
    print("--------------------------------------------------------")
    print()
    ErrorRate(dfcol,predclassfifty)
    
main()