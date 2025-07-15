# 📊 AI Data Analysis Project

A data analysis project utilizing machine learning techniques for feature selection, clustering, and classification on global development data.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-orange.svg)

## 🎯 Objective

This project analyzes global development indicators to:

* **Clean** and preprocess data (handle missing values, correlations)
* **Classify** countries into economic categories (wealthy, average, poor)
* **Group** countries by similarity using various clustering algorithms

## 📈 Data

**Source**: [World Bank Open Data](http://databank.worldbank.org)

* **Indicators**: GDP, life expectancy, education, health, etc.
* **Countries**: Worldwide country-level data
* **Period**: Selected reference year

## 🔬 Implemented Methods

### Preprocessing

* Handling missing values
* Removing correlated features
* Data normalization
* KNN imputation

### Clustering

* **K-means**: Partitioning data into k groups
* **Hierarchical Clustering**: Dendrogram creation and grouping
* **DBSCAN**: Density-based clustering

### Classification

* **Decision Tree**: Supervised classification
* **Metrics**: F1-score, accuracy, recall

## 🛠️ Features

* Automatic data cleaning
* Correlation-based feature selection
* Multiple clustering algorithms
* Supervised classification
* Performance metrics evaluation
* Integrated visualizations

## 📝 License

MIT License - See `LICENSE` for details.
