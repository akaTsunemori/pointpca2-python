# pointpca2-python
#### Cross-language replication and analysis of "pointpca2" from MATLAB to Python.
This project aims to adapt the [pointpca2](https://github.com/cwi-dis/pointpca2/) project (2023 Grand Challenge on Objective Quality Metrics for Volumetric Contents), originally written in MATLAB, to Python. The goal is to replicate its functionality in a different programming ecosystem, ensuring that features generated in both environments are comparable and interchangeable for further analysis. Upon successful replication, the project will proceed to utilize these features for regression analysis against a dataset's subjective scores. A comprehensive comparison of the performance of regressors, facilitated through Pearson and Spearman correlation coefficients, will be carried out for each version of the code. Furthermore, a statistical t-test will be conducted to rigorously compare the correlation results derived from both MATLAB and Python implementations, ensuring the validity and reliability of the adaptation process.


# Key Objectives
1. Code Adaptation: Convert the "pointpca2" project code from MATLAB to Python, ensuring that the core functionality and output remain consistent across both languages.
2. Feature Generation: Generate PCA features using both the original MATLAB code and the newly developed Python code. We will validate the equivalence of these features through statistical methods to ensure that both implementations produce comparable results.
3. Regression Analysis: Use the generated features to perform regression analysis against subjective scores in the dataset. This will involve using a set of regression models and fitting them with the features to predict the subjective scores. Calculate both Pearson and Spearman correlation coefficients for each regressor.
4. Correlation Analysis: Plot the correlation coefficients for each regressor. This analysis will provide insights into the linear and rank-order relationships between predicted scores and actual subjective scores.
5. Statistical Comparison: Conduct a t-test to statistically compare the correlation coefficients obtained from MATLAB and Python implementations. This step is critical to assess whether the differences in correlations (if any) are statistically significant, providing a quantitative measure of the adaptation's fidelity.

# Expected Outcomes
- A fully functional Python version of the "pointpca2" MATLAB project, verified for accuracy and equivalence.
- A detailed comparison of regression model performances using PCA features from both MATLAB and Python implementations.
- A statistical analysis report providing evidence on the equivalence (or differences) in correlation coefficients derived from both languages' codes.

# Prerequisites
## General
- anaconda3 ([https://www.anaconda.com/](https://www.anaconda.com/))
## For MATLAB feature generation
- MATLAB (version R2023a tested)
- MATLAB Engine API for Python ([https://pypi.org/project/matlabengine/](https://pypi.org/project/matlabengine/))

# Installation
```bash
# Clone the repository and navigate into it
git clone https://github.com/akaTsunemori/pointpca2-python.git
cd pointpca2-python

# Set up the conda environment
conda env update --file environment.yml

# Activate the environment
conda activate pointpca2-python
```

# Usage
The files in this section are located in the [src](src) directory.

## pointpca2.py
This is the project's main module. It replicates all the functions present in pointpca2's original code. The main function, pointpca2, should be called with the path for the reference and the path for the test point clouds, it returns an array consisting of the generated features. Additionally, an optional "decimation_factor" argument can be passed, where a decimation factor of 2 would mean that both the reference and test point clouds will be halved.

## generate_features_dataset.py
Generates tables containing dataset features. It assumes the dataset CSV format as follows:
  
|SIGNAL  |REF     |SCORE   |LOCATION|REFLOCATION|ATTACK  |CLASS   |
|--------|--------|--------|--------|-----------|--------|--------|

The script expects correct location paths. It skips rows where exceptions occur during pointpca2 computations.

Checkpoints are saved in **./features**. Do not modify these files unless the dataset is fully processed, as this could compromise the checkpoint system. Each checkpoint also has a **.bak** file for data safety.

CLI arguments:
- dataset_name: Name of the dataset, e.g., **APSIPA**
- dataset_csv: Path to the dataset CSV, e.g., **/home/user/Documents/APSIPA/apsipa.csv**
- decimation_factor: the decimation factor, e.g., **4**

You can process multiple datasets concurrently by running this script with different parameters. Outputs are saved in the "features" folder.

## generate_features_matlab.py
Use this script to generate features using the original MATLAB version of pointpca2. This script functions similarly to generate_features_dataset.py but includes an extra command-line argument to specify the path to the MATLAB version of pointpca2. Please note that decimation is not supported in this script.

CLI arguments:
- dataset_name: Name of the dataset, e.g., **APSIPA**
- dataset_csv: Path to the dataset CSV, e.g., **/home/user/Documents/APSIPA/apsipa.csv**
- pointpca2_path: Path to the original pointpca2 MATLAB code/repository, e.g., **/home/user/Documents/pointpca2/**

Although possible, it is not recommended to run this script concurrently as it's very RAM hungry. Outputs are saved in the "features" folder.

## regressions.py
Performs regressions using models from LazyPredict based on tables generated by **generate_features_dataset.py**, applying Leave One Group Out and Group K-Fold techniques.

CLI arguments:
- csv_path_reference: path to the reference feature table, e.g., **./results/APSIPA_pointpca2_NoDecimation.csv**
- csv_path_test: path to the reference feature table, e.g., **./results/APSIPA_pointpca2_DecimateBy2.csv**
- dataset_name: the name of the dataset. Example: **APSIPA**

Outputs are saved in the "regressions" folder.

## plots.py
Plots Pearson and Spearman correlation coefficients for each regressor using tables from **regressions.py**.

CLI arguments:
- Paths to the reference and test regression tables for Leave One Group Out and Group K-Fold techniques:
  - csv_path_regression_reference_LeaveOneGroupOut
  - csv_path_regression_reference_GroupKFold
  - csv_path_regression_test_LeaveOneGroupOut
  - csv_path_regression_test_GroupKFold
- dataset_name: Name of the dataset, e.g., **APSIPA**.

Outputs are saved in the "plots" folder.
  
## ttests.py
Conducts a t-test to statistically compare correlation coefficients from MATLAB and Python regressions, assessing the adaptation's fidelity.

CLI arguments:
- Paths to the reference and test regression tables for Leave One Group Out and Group K-Fold techniques:
  - csv_path_regression_reference_LeaveOneGroupOut
  - csv_path_regression_reference_GroupKFold
  - csv_path_regression_test_LeaveOneGroupOut
  - csv_path_regression_test_GroupKFold
- dataset_name: Name of the dataset, e.g., **APSIPA**.
 
Outputs are saved in the "ttests" folder.

# Results
Results for various datasets are in the [results](results) folder, including checkpoints, feature tables, plots, regressions, and t-tests. APSIPA dataset results are exemplified below.

## Leave One Group Out
<details>
  <summary>Spoiler</summary>

![APSIPA_LeaveOneGroupOut](results/APSIPA/plots/APSIPA_LeaveOneGroupOut.png)
|Model                        |p-value (Pearson)|p_value <= 0.05 (Pearson)|p-value (Spearman)|p_value <= 0.05 (Spearman)|
|-----------------------------|-----------------|-------------------------|------------------|--------------------------|
|AdaBoostRegressor            |0.59             |False                    |0.58              |False                     |
|BaggingRegressor             |0.87             |False                    |0.48              |False                     |
|BayesianRidge                |0.90             |False                    |0.90              |False                     |
|DecisionTreeRegressor        |0.37             |False                    |0.35              |False                     |
|ElasticNet                   |1.00             |False                    |0.99              |False                     |
|ElasticNetCV                 |0.87             |False                    |0.94              |False                     |
|ExtraTreeRegressor           |0.59             |False                    |0.60              |False                     |
|ExtraTreesRegressor          |0.92             |False                    |0.91              |False                     |
|GammaRegressor               |1.00             |False                    |0.98              |False                     |
|GaussianProcessRegressor     |1.00             |False                    |0.99              |False                     |
|GradientBoostingRegressor    |0.78             |False                    |1.00              |False                     |
|HistGradientBoostingRegressor|0.74             |False                    |0.79              |False                     |
|HuberRegressor               |0.92             |False                    |1.00              |False                     |
|KNeighborsRegressor          |1.00             |False                    |0.99              |False                     |
|KernelRidge                  |0.89             |False                    |0.98              |False                     |
|LGBMRegressor                |0.88             |False                    |0.88              |False                     |
|Lars                         |0.38             |False                    |0.33              |False                     |
|LarsCV                       |0.32             |False                    |0.30              |False                     |
|Lasso                        |1.00             |False                    |0.99              |False                     |
|LassoCV                      |0.95             |False                    |0.95              |False                     |
|LassoLars                    |1.00             |False                    |0.99              |False                     |
|LassoLarsCV                  |0.96             |False                    |0.92              |False                     |
|LassoLarsIC                  |0.86             |False                    |0.92              |False                     |
|LinearRegression             |0.89             |False                    |0.96              |False                     |
|LinearSVR                    |0.80             |False                    |0.97              |False                     |
|MLPRegressor                 |0.99             |False                    |0.92              |False                     |
|NuSVR                        |0.99             |False                    |0.98              |False                     |
|OrthogonalMatchingPursuit    |0.99             |False                    |0.98              |False                     |
|OrthogonalMatchingPursuitCV  |1.00             |False                    |0.97              |False                     |
|PassiveAggressiveRegressor   |1.00             |False                    |0.93              |False                     |
|PoissonRegressor             |0.99             |False                    |0.99              |False                     |
|RANSACRegressor              |0.33             |False                    |0.52              |False                     |
|RandomForestRegressor        |0.98             |False                    |0.90              |False                     |
|Ridge                        |0.89             |False                    |0.98              |False                     |
|RidgeCV                      |0.96             |False                    |0.94              |False                     |
|SGDRegressor                 |1.00             |False                    |1.00              |False                     |
|SVR                          |1.00             |False                    |0.99              |False                     |
|TransformedTargetRegressor   |0.89             |False                    |0.96              |False                     |
|TweedieRegressor             |1.00             |False                    |1.00              |False                     |
|XGBRegressor                 |1.00             |False                    |0.89              |False                     |

*p-values rounded to 2 decimal places for better visibility*

</details>

## Group K-Fold
<details>
  <summary>Spoiler</summary>

![APSIPA_GroupKFold](results/APSIPA/plots/APSIPA_GroupKFold.png)
|Model                        |p-value (Pearson)|p_value <= 0.05 (Pearson)|p-value (Spearman)|p_value <= 0.05 (Spearman)|
|-----------------------------|-----------------|-------------------------|------------------|--------------------------|
|AdaBoostRegressor            |0.59             |False                    |0.58              |False                     |
|BaggingRegressor             |0.87             |False                    |0.48              |False                     |
|BayesianRidge                |0.90             |False                    |0.90              |False                     |
|DecisionTreeRegressor        |0.37             |False                    |0.35              |False                     |
|ElasticNet                   |1.00             |False                    |0.99              |False                     |
|ElasticNetCV                 |0.87             |False                    |0.94              |False                     |
|ExtraTreeRegressor           |0.59             |False                    |0.60              |False                     |
|ExtraTreesRegressor          |0.92             |False                    |0.91              |False                     |
|GammaRegressor               |1.00             |False                    |0.98              |False                     |
|GaussianProcessRegressor     |1.00             |False                    |0.99              |False                     |
|GradientBoostingRegressor    |0.78             |False                    |1.00              |False                     |
|HistGradientBoostingRegressor|0.74             |False                    |0.79              |False                     |
|HuberRegressor               |0.92             |False                    |1.00              |False                     |
|KNeighborsRegressor          |1.00             |False                    |0.99              |False                     |
|KernelRidge                  |0.89             |False                    |0.98              |False                     |
|LGBMRegressor                |0.88             |False                    |0.88              |False                     |
|Lars                         |0.38             |False                    |0.33              |False                     |
|LarsCV                       |0.32             |False                    |0.30              |False                     |
|Lasso                        |1.00             |False                    |0.99              |False                     |
|LassoCV                      |0.95             |False                    |0.95              |False                     |
|LassoLars                    |1.00             |False                    |0.99              |False                     |
|LassoLarsCV                  |0.96             |False                    |0.92              |False                     |
|LassoLarsIC                  |0.86             |False                    |0.92              |False                     |
|LinearRegression             |0.89             |False                    |0.96              |False                     |
|LinearSVR                    |0.80             |False                    |0.97              |False                     |
|MLPRegressor                 |0.99             |False                    |0.92              |False                     |
|NuSVR                        |0.99             |False                    |0.98              |False                     |
|OrthogonalMatchingPursuit    |0.99             |False                    |0.98              |False                     |
|OrthogonalMatchingPursuitCV  |1.00             |False                    |0.97              |False                     |
|PassiveAggressiveRegressor   |1.00             |False                    |0.93              |False                     |
|PoissonRegressor             |0.99             |False                    |0.99              |False                     |
|RANSACRegressor              |0.33             |False                    |0.52              |False                     |
|RandomForestRegressor        |0.98             |False                    |0.90              |False                     |
|Ridge                        |0.89             |False                    |0.98              |False                     |
|RidgeCV                      |0.96             |False                    |0.94              |False                     |
|SGDRegressor                 |1.00             |False                    |1.00              |False                     |
|SVR                          |1.00             |False                    |0.99              |False                     |
|TransformedTargetRegressor   |0.89             |False                    |0.96              |False                     |
|TweedieRegressor             |1.00             |False                    |1.00              |False                     |
|XGBRegressor                 |1.00             |False                    |0.89              |False                     |

*p-values rounded to 2 decimal places for visibility*

</details>

# Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. You can simply open an issue with the tag "enhancement".

# Acknowledgments
- [pointpca2](https://github.com/cwi-dis/pointpca2/)
- [Anaconda](https://www.anaconda.com/)
- [LazyPredict](https://lazypredict.readthedocs.io/en/latest/)
- [NumPy](https://numpy.org/doc/stable/)
- [Pandas](https://pandas.pydata.org/docs/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [MATLAB Engine API for Python](https://pypi.org/project/matlabengine/)

# License
Licensed under the BSD 3-Clause Clear License

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)
