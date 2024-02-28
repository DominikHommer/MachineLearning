# Linear Models in Machine Learning

This directory, "Linear_Models," within the "MachineLearning" repository, focuses on exploring various linear models through practical exercises. It includes three Jupyter Notebooks and two datasets aimed at providing insights into different machine learning techniques involving linear regression, PCA with Ridge and Lasso regression, and exponential regression.

## Notebooks Overview

- **linear_regression.ipynb:** Explores the fundamentals of linear regression using the "advertising.csv" dataset. This notebook demonstrates how advertising expenditures impact sales outcomes.

- **fuel_efficiency_pca_ridge_lasso.ipynb:** Applies Principal Component Analysis (PCA) followed by Ridge and Lasso regression on the "auto.csv" dataset to predict fuel efficiency.

- **exp_regression.ipynb:** Utilizes exponential regression to model the annual world population growth based on data scraped from [Worldometers](https://www.worldometers.info/world-population/world-population-by-year/).

## Datasets

- **auto.csv:** Contains data on various car attributes, including fuel efficiency. Sourced from [Kaggle](https://www.kaggle.com/datasets/mandragorassster/autocsv), it is used in the "fuel_efficiency_pca_ridge_lasso.ipynb" notebook.

- **advertising.csv:** Features data on advertising expenditures across TV, radio, and newspaper mediums, along with sales figures. It is utilized in the "linear_regression.ipynb" notebook and can be found [here on Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv).

## Getting Started

To explore these notebooks, follow the instructions below:

### Prerequisites

Ensure you have the following tools and libraries installed:
- Python 3.x
- Jupyter Notebook or JupyterLab
- Required Python packages: pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn

### Installation

Clone the MachineLearning repository and navigate to the "Linear_Models" directory:

```bash
git clone https://github.com/DominikHommer/MachineLearning.git
cd MachineLearning/Linear_Models
```

Install the necessary Python packages:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### Running the Notebooks

Launch Jupyter Notebook or JupyterLab and open the desired notebook:

```bash
jupyter notebook
```

## Analysis Overview and Results

Each notebook provides a detailed analysis, from data preprocessing to model evaluation:

- The linear regression notebook reveals how TV advertising spend positively correlates with sales.
- The PCA with Ridge and Lasso regression notebook offers insights into feature importance and regularization techniques in predicting fuel efficiency.
- The exponential regression notebook models the growth of the world's population over the years, demonstrating the application of non-linear models to real-world data.

## License

This project is licensed under the MIT License[LICENSE.md](LICENSE.md)

