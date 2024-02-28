# Classification Projects

This subdirectory, "Classification", within the "MachineLearning" repository, includes projects focusing on classification tasks using various machine learning algorithms. It contains two Jupyter Notebooks and associated datasets aimed at demonstrating classification techniques on different types of data: fruit classification and spam email detection.

## Notebooks Overview

- **fruit_classification_pca_crossval_kNN.ipynb:** This notebook demonstrates the process of classifying different types of fruits using the k-Nearest Neighbors (kNN) algorithm. It includes steps for data preprocessing, dimensionality reduction using Principal Component Analysis (PCA), and model validation through cross-validation.

- **svm_spammail.ipynb:** Focuses on detecting spam emails using the Support Vector Machine (SVM) algorithm. It showcases the effectiveness of SVM in text classification tasks, starting with an initial dataset for training and then validating the model's generalization capability on a completely new dataset.

## Datasets

The notebooks utilize the following datasets:

- **spam_ham_dataset.csv:** Used in the `svm_spammail.ipynb` notebook for initial model training. Contains emails labeled as spam or ham (non-spam).
  [Spam Mails Dataset on Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data)

- **emails.csv:** Employed in the `svm_spammail.ipynb` notebook for model validation. Serves to test the model's generalization on unseen data.
  [Spam Email Dataset on Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)

- **fruit.csv:** Used in the `fruit_classification_pca_crossval_kNN.ipynb` notebook for fruit classification tasks. Includes data on various fruit types and their characteristics.
  [Fruit Dataset on Kaggle](https://www.kaggle.com/datasets/wuxi5791/fruitcsv)

## Getting Started

To explore these notebooks, ensure you have Jupyter Notebook or JupyterLab installed, along with necessary Python libraries such as pandas, numpy, matplotlib, seaborn, scikit-learn, and others as required by specific notebooks.

### Installation

Clone the MachineLearning repository and navigate to the "Classification" directory:

```bash
git clone https://github.com/your-username/MachineLearning.git
cd MachineLearning/Classification
```

Install the necessary Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebooks

Launch Jupyter Notebook or JupyterLab and open the desired notebook:

```bash
jupyter notebook
```

## Reflections

The classification projects were initiated to explore and demonstrate the application of machine learning algorithms in different classification tasks. The strong performance of models, especially in spam email detection with SVM, highlighted the potential of these algorithms for practical applications. Testing the SVM model on a completely new dataset and achieving impressive results underscored the models' robustness and adaptability.

