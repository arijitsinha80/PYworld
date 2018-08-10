Dataset - 
Download the file from [link](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/') -  'breast-cancer-wisconsin.data

Dataset Info - It has the columns as 

'ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 
 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 
'Class'. 

Upon investigation, Column A7 (Bare Nuclei) has 16 Null Values for ID's  - 
          1057013, 1096800, 1183246, 1184840, 1193683, 1197510, 1241232, 169356, 432809, 563649
          606140, 61634, 704168, 733639 ,1238464 ,1057067          

Data Processing - Impute/ Replace the null values with Mean of 3.54

Plotted Histogram for each of the columns and correlation Matrix.

Gather Statistical Information for Standard Deviation, Mean, Variance and Median.

Execute the script - 
          There is only 1 script as "main.py" can be executed to generate all the results. 

This script downloads the data from weblink mentioned above and process the data. This script has multiple functions - contblk(x), repbnk(x), statdf(x), histplt(x), corrmatrix(x), for finding the null values, replace null values, calculate statistics, ploting histogram and plotting correlation matrix.

Also, Created the functions as - 
          picktwocor(x), assigndf(x, u1, u2), updatemean(x, pc), recalmeanfifty(x, col, pc) for
          picking the two random means and then creating and assigninig the prediction for cluster 2 and 4. 
          Once we have initial clustering 2 and 4 with initial random means, update the mean with from above cluster2 and 4, repeat the           same for 5000 iterations or if matched with previous iterations.

After the run submitted the screenshot of comparision for class and predicted class for first 20 ID's.

Once we have the overall prediction on 699 obeservations, we have taken the comparision for Cluster 2 and 4.
          1. Not Matching Benign: 13
          2. Not Matching Malignant 11
          3. Total number of Benign: 456
          4. Total number of Malignant: 243
          5. Total numberof Records 699
          6. Total Not Matched: 24
          7. Error with Benign : 0.02851
          8. Error with Malignant : 0.04527
          9. Total Error Rate : 0.03433
