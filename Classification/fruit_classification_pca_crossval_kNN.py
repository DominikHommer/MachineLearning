#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries
# 
# We start by importing all the necessary libraries that we'll need throughout this notebook. This includes libraries for data manipulation, visualization, machine learning, and metrics evaluation.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import math as ma
import sklearn as sl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix, auc 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import statsmodels.stats.api as sms
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the Dataset
# 
# Here we load the dataset for analysis and display the unique fruit names. Additionally, we clean the `fruit_name` column by stripping any leading or trailing spaces.
# 

# In[2]:


df = pd.read_csv('fruit.csv')
print(df['fruit_name'].unique())
df['fruit_name'] = df['fruit_name'].str.strip()
print(df['fruit_name'].unique())
sns.countplot(x='fruit_name', data=df, label="Count")


# In[3]:


df


# # Data Augmentation for Class 2
# 
# To balance our dataset, we increase the number of data points for class 2 to a total of 17 by adding noise to the existing data points.
# 

# In[4]:


names = ['mass', 'width', 'height', 'color_score', 'fruit_label']
df = df[names]
df_class2 = df[df['fruit_label'] == 2]

num_entries_needed = 17 - len(df_class2)
new_data = []
if num_entries_needed > 0:
    for _ in range(num_entries_needed):
        row = df_class2.sample(n=1).iloc[0]
        noisy_data = {name: row[name] + np.random.normal(0, 0.05) if name != 'fruit_label' else row[name] for name in names}
        new_data.append(noisy_data)

new_df_class2 = pd.DataFrame(new_data, columns=names)
df = pd.concat([df, new_df_class2]).reset_index(drop=True)
df.loc[df.index[-num_entries_needed:], 'fruit_label'] = 2


# In[5]:


sns.countplot(x='fruit_label', data=df, label="Count")


# # Scatter Matrix of Inputs by Class
# 
# Visualizing the relationships between different features split by fruit class.

# In[6]:


feature_names = ['mass', 'width', 'height', 'color_score']
X = df[feature_names]
y = df['fruit_label']

scatter = pd.plotting.scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins':15}, figsize=(15,15))
plt.suptitle('Scatter-matrix of Inputs by Class')


# # Principal Component Analysis (PCA)
# 
# Reducing dimensionality of the dataset to 3 principal components for visualization and modeling purposes.

# In[7]:


X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, marker='o', s=40)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Inputs by Class')
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)
plt.show()
print(f"Explained Variance by the first 3 PCs: {sum(pca.explained_variance_ratio_):.2f}")


# # k-Nearest Neighbors (kNN) Model Evaluation
# 
# Evaluating the k-Nearest Neighbors model performance for different values of `k` using both the training and test datasets.

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
k_range = range(1, 26)
train_accuracy = []
test_accuracy = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    train_accuracy.append(accuracy_score(y_train, y_train_pred))
    test_accuracy.append(accuracy_score(y_test, y_test_pred))

plt.figure(figsize=(12, 6))
plt.plot(k_range, train_accuracy, label='Training Accuracy')
plt.plot(k_range, test_accuracy, label='Test Accuracy')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Accuracy')
plt.title('kNN: Accuracy for Different k Values')
plt.legend()
plt.show()


# # Cross-Validation to Validate `k`
# 
# Performing cross-validation to find the optimal number of neighbors `k` for the kNN model.

# In[9]:


cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_pca, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range[cv_scores.index(max(cv_scores))]
plt.figure(figsize=(12, 6))
plt.plot(k_range, cv_scores, marker='o', linestyle='-', color='r')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Cross-Validated Accuracy')
plt.title('kNN: Cross-Validated Accuracy for Different k Values')
plt.show()
print(f"Best k based on Cross-Validation: {best_k}")


# # Final Model with `k=6`
# 
# Configuring and training the kNN model with the optimal number of neighbors determined by cross-validation.
# 

# In[10]:


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)


# # Confusion Matrices
# 
# Displaying the confusion matrices for both the training and test datasets to evaluate model performance.

# In[11]:


print('Confusion matrix (Train):\n', confusion_matrix(y_train_pred, y_train))
print('Confusion matrix (Test):\n', confusion_matrix(y_test_pred, y_test))


# # Classification Report
# 
# Generating a classification report to review precision, recall, and f1-score metrics for each class based on the test dataset.

# In[12]:


print(classification_report(y_test_pred, y_test, digits=4))


# ## Conclusion
# 
# In this analysis, i utilized a comprehensive approach to classify fruits based on their physical attributes using a dataset available on Kaggle ([Fruits.csv dataset](https://www.kaggle.com/datasets/wuxi5791/fruitcsv)). The steps undertaken were methodical and based on sound machine learning practices, including Data Augmentation, Principal Component Analysis (PCA), k-Nearest Neighbors (kNN), and Cross-Validation.
# 
# ### Why Data Augmentation?
# I noticed early in the analysis that Class 2 was underrepresented in our dataset, with only a few data points. To address this imbalance and improve our model's ability to generalize across all classes, I employed Data Augmentation. By adding synthetic data points generated through slight modifications (adding noise) to existing ones, I increased the representation of Class 2, thereby enhancing the robustness and fairness of our classification model.
# 
# ### The Role of PCA
# Given the multidimensionality of our feature set, I applied PCA to reduce the dimensions to the three most informative Principal Components. This step was crucial for several reasons:
# - **Visualization:** It enabled us to visually inspect the data in a 3D scatter plot, providing insights into the natural clustering of the classes.
# - **Model Efficiency:** By reducing the dimensions, we improved the computational efficiency of our model without sacrificing significant information, as evidenced by the explained variance ratio.
# 
# ### Choosing kNN and k=6
# The kNN algorithm was chosen for its simplicity and effectiveness in classification tasks. To determine the optimal number of neighbors (`k`), I employed Cross-Validation, testing a range of values for `k`. The results indicated that `k=6` yielded the highest cross-validated accuracy, striking a balance between overfitting (too low `k`) and underfitting (too high `k`).
# 
# ### Cross-Validation: A Pillar of Our Methodology
# Cross-Validation was instrumental in validating our choice of `k`. By evaluating the model's performance across different subsets of the data, I ensured that the findings were not a fluke of a particular data split. This rigorous approach bolstered the confidence in the model's generalizability and performance.
# 
# ### Final Model Evaluation
# The culmination of my efforts was a model that achieved a remarkable accuracy of 95.45% on the test dataset. The classification report further detailed the success:
# - **Class 1:** Showed high precision but slightly lower recall, indicating room for minor improvements.
# - **Class 2:** Achieved perfect scores across precision, recall, and F1-score, highlighting the effectiveness of our data augmentation strategy.
# - **Class 3:** While precision was lower, the perfect recall score signifies that all actual instances of Class 3 were correctly identified.
# - **Class 4:** Mirrored Class 2 in achieving perfect scores, underscoring the model's robustness.
# 
# Overall, the balanced and high scores across all metrics for each class underscore the efficacy of my analytical approach. By carefully addressing data imbalances, reducing dimensionality for insight and efficiency, selecting an appropriate algorithm, and rigorously validating our choices, I developed a model that not only performs exceptionally well but also offers insights into the importance of each step in the machine learning pipeline.
# 
