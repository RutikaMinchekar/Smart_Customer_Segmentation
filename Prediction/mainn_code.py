# Generated from: project-ml1.ipynb
# Converted at: 2026-01-08T15:26:56.167Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # Smart Customer Segmentation


# **Problem Statement** Businesses face challenges in marketing due to a lack of customer segmentation. Without understanding different spending behaviors, marketing efforts become ineffective. This project will address that by using machine learning to group customers, allowing businesses to personalize marketing, reduce costs, and increase engagement.


# # 1.Business Goal of project


# The goal of this project is to use machine learning to segment customers based on their income and spending habits, helping businesses create targeted marketing strategies to improve sales, customer retention, and resource efficiency.


# Importing Libraries


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


#  # 2. Data Understanding


# Data Loading -> data loading is the foundation for machine learning workflows. 


customer_data=pd.read_csv("C:/Users/Admin/Downloads/clustering_customer/Mall_Customers.csv")

customer_data

# ## EDA operation 


#Return first 5 line of data from dataset
customer_data.head()

#Return last 5 line of data from dataset
customer_data.tail()

#Return colums name and ther datatype or having any missing value of data from dataset
customer_data.info()

#Return total count of total no. rows and columns of data from dataset
customer_data.shape

#Checking either our dataset have any missing value
customer_data.isnull().sum()

#Return math info. of dataset like mean,mod SD etc
customer_data.describe()

#Returns the data type of colums
customer_data.dtypes

# # 3.Data Preparation
#    
# 1. **Data Cleaning:** No missing values, handled outliers
# 2. **Machine Learning Task:** K-Means clustering, where there is no                  explicit target variable.Independent variables include customer features        like CustomerID,Age,Gener AnuualIncome.
# 3. **Train-Test Split:** Not applicable for clustering;
#      the whole dataset is used for forming clusters, and evaluation is               typically done using cluster validation metrics rather than splitting           the data.
# 


# # 4.Data Preprocessing


# 4.1 Feature Engineering - transformed existing ones (e.g., grouping age into ranges).

print(customer_data)

#finding minimum and maximum age from (Age column)
min_age =customer_data['Age'].min()

print(min_age)

max_age =customer_data['Age'].max()

print(max_age)

# The minimum age is 18 and the maximum age is 70, and want to group the ages into appropriate age ranges (bins), you can define the bins based on logical or common groupings, such as:
# 
#     18-30 (Young adults)
#     31-40 (Adults)
#     41-50 (Middle-aged adults)
#     51-60 (Older adults)
#     61-70 (Seniors)


bins = [18, 30, 40, 50, 60, 70]  # Bin edges- defines the age groups.
labels = ['18-30', '31-40', '41-50', '51-60', '61-70']  # Corresponding labels- gives each bin a human-readable name.


#pd.cut() is used to categorize each age into its respective group.
customer_data['Age Group'] = pd.cut(customer_data['Age'], bins=bins, labels=labels, right=False)
print(customer_data)


print(customer_data[['Age', 'Age Group']])

# Group by 'Age Group' and count the total number of people in each group
Age_group_count = customer_data.groupby('Age Group').size()

print(Age_group_count)

# 4.2 Data Normalization - Label Encoding to rescale 'Genre'cloumn in 0 & 1 range
from sklearn.preprocessing import LabelEncoder

#Basically create object and initilize the LabelEncoder
labelencoder=LabelEncoder()

customer_data['Genre']=labelencoder.fit_transform(customer_data['Genre'])

print(customer_data.iloc[:, 0:6])

print(customer_data.head(5));

# By LabelEncoder,
# 
# Catogorical data converts into numerical by setting Male=1,Female=0;


# # 5.Data Visualization(EDA)


# Data visualization is the graphical representation of data to help communicate information clearly and efficiently.
# 
# **Bar Charts:** Compare categorical data by displaying rectangular bars. The length of each bar represents the frequency of each category (e.g., distribution of age groups).


import matplotlib.pyplot as plt

# Plotting bar chart for the 'Age Group' column (categorical)
plt.figure(figsize=(4,4))
customer_data['Age Group'].value_counts().plot(kind='bar', color='green', edgecolor='black')
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Gives a visual overview of how many customers fall into each predefined age group.


#Basically we need these both colums, so we just take this col values by using iloc(index Location)
X=customer_data.iloc[:,[3,4]].values

# Choosing the correct no. of cluster


print(X)

# WCSS-> Within Clusters Sum of Square


# Here we finding Centroid value to find cluster value using kmeans algo


# ### Training and testing data


#Finding WCSS value for different number of cluster
wcss=[]#list

for i in range(1,11):  #(n-1=> 11-1=10)
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    

#Plot an Elbow grapgh
sns.set()
plt.plot(range(1,11),wcss)
plt.title("The Elobow Point Graph")
plt.xlabel("Number of Cluster- Annual Income")
plt.ylabel("Wcss-Spending")
plt.show()

# **Elbow method**
# The elbow method helps us find the best number of clusters by looking at where the curve bends the most. In your graph, the biggest bend happens at 5 clusters. 


# # 6. Model Selection 
# 


# 
# 1. **Model Choice**: K-Means is used to group customers into clusters based on similar characteristics (e.g., spending behavior, demographics).
# 2. **Process**: Data is scaled, K is chosen using methods like the Elbow Method, and the model is trained to find distinct customer segments.
# 3. **Use**: These segments help create targeted marketing strategies or personalized offers.
# 


# # Algorithm - kmeans clustering


# K-Means is a popular clustering algorithm used in machine learning for unsupervised learning. Its main goal is to partition data into K clusters, where each cluster contains data points that are similar to each other.


kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)



#Return a label for each  data point there cluster
Y = kmeans.fit_predict(X)

print(Y)

import joblib

joblib.dump(kmeans, "kmeans_model.pkl",compress=3)
print("✅ kmeans_model.pkl saved")


#Ploting the clusters and their centroid
plt.figure(figsize=(6,4))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='cluster 0')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='cluster 1')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='cluster 2')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='cluster 3')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='brown',label='cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=90,c='cyan',label='Centroids')

plt.title("Customer Group")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# Clutser=5(0-5)
# so Labeling like
# 
# 1. Cluster 0:  High Income, Low Spending.
# 2. Cluster 1:  Medium Income, High Spending.
# 3. Cluster 2:  High Income, High Spending.
# 4. Cluster 3:  Low Income, High Spending.
# 5. Cluster 4:  Low Income, Low Spending.
# 
# 


# # 7. Prediction
# 


Prediction=kmeans.predict([[15,81]])

print(Prediction)

Prediction=kmeans.predict([[137,83]])

print(Prediction)

Prediction=kmeans.predict([[42,52]])

print(Prediction)

Prediction=kmeans.predict([[16,6]])

print(Prediction)

Prediction=kmeans.predict([[99,97]])

print(Prediction)

# By observing the cluster this is the result 
# 
#  
# 1. Cluster 0 (High Income, Low Spending): Target with exclusive investment products or premium savings plans.
# 
#    Example: High-end laptops, premium gadgets like Apple products, investment plans, smart home devices.
# 
# 


# 2. Cluster 1 (Medium Income, High Spending): Focus on lifestyle products and frequent shopper discounts.
# 
#     Example: Fashion apparel, fitness equipment, smartwatches, beauty and grooming products.


# 3.Cluster 2 (High Income, High Spending): Market luxury items, exclusive offers, and premium services.
# 
# Example: Luxury watches, designer clothing, premium smartphones (like iPhone, Samsung Galaxy), high-end home appliances.
# 


# 4.Cluster 2 (High Income, High Spending): Market luxury items, exclusive offers, and premium services.
# 
# Example: Affordable smartphones, budget-friendly clothing, electronics on EMI, trendy accessories.


# 5.Cluster 4 (Low Income, Medium Spending): Target with budget-friendly products and loyalty discounts for repeat purchases
# 
# Example: Economy smartphones, home essentials, affordable kitchen appliances, budget footwear.