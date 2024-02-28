#!/usr/bin/env python
# coding: utf-8

# # SVM Model for Spam Email Detection
# 
# I initially embarked on this project to experiment with the application of Support Vector Machines (SVM) in the context of email spam detection. Surprisingly, the results were impressively strong right off the bat. The process involved training an SVM model on a dataset and then validating its performance on an entirely new dataset to test its generalization capabilities.
# 

# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the Initial Dataset
# 
# The first dataset, sourced from Kaggle, contains various emails labeled as spam or ham. This dataset was used to train the initial SVM model.
# 
# [Spam Mails Dataset on Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data)
# 

# In[33]:


df = pd.read_csv('spam_ham_dataset.csv')
df.head()


# ## Preprocessing and Model Training
# 
# I split the dataset into training and testing sets, applied TF-IDF vectorization to convert the email texts into numerical features, and trained an SVM model with a linear kernel.
# 

# In[34]:


X = df['text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectors, y_train)
y_pred = svm_model.predict(X_test_vectors)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ## Model Optimization with GridSearchCV
# 
# To further refine the model, I utilized GridSearchCV to find the optimal hyperparameters for the SVM model.
# 

# In[35]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_vectors, y_train)


# ## Evaluating the Optimized Model
# 
# After finding the best hyperparameters, I evaluated the optimized model's performance on the test set.
# 

# In[36]:


print("Best Hyperparameters:", grid.best_params_)
print("Best Average Accuracy:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_vectors)

print("Accuracy of the Best Model:", accuracy_score(y_test, y_pred))
print("Classification Report of the Best Model:\n", classification_report(y_test, y_pred))


# ## Testing the Model on New Data
# 
# Curious about the model's generalization capability, I decided to test it on a completely different dataset, also from Kaggle, to see how well it could adapt to unseen data.
# 
# [Spam Email Dataset on Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)

# In[37]:


val_df = pd.read_csv('emails.csv')
X_val_df = val_df['text']
y_val_df = val_df['spam']

X_val_transformed = vectorizer.transform(X_val_df)
y_pred_val = best_model.predict(X_val_transformed)

accuracy_val = accuracy_score(y_val_df, y_pred_val)
print("Accuracy on the Validation Dataset:", accuracy_val)
print("Classification Report for the Validation Dataset:\n", classification_report(y_val_df, y_pred_val))


# ## Reflections
# 
# Initially, I just wanted to test how an SVM model would perform on email spam detection. To my surprise, not only did the model demonstrate strong results on the training and test sets, but it also generalized exceptionally well to an entirely new dataset. These outcomes highlight the robustness and versatility of SVMs in handling text classification tasks, even in the face of varied data.
# 
