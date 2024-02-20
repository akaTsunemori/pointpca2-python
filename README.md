<h1 align="center">
  pointpca2-python
</h1>

#### Cross-language replication and analysis of "pointpca2" using MATLAB and Python.

This project focuses on the adaptation of the [pointpca2](https://github.com/cwi-dis/pointpca2/) project (2023 Grand Challenge on Objective Quality Metrics for Volumetric Contents), written in MATLAB, into Python, aiming to replicate its functionality in a different programming ecosystem. The primary objective is to ensure that features generated from both environments are comparable and can be used interchangeably for further analysis. Upon successful replication, the project will proceed to utilize these features for regression analysis against a dataset's subjective scores. A comprehensive comparison of the performance of regressors, facilitated through Pearson and Spearman correlation coefficients, will be carried out for each version of the code. Furthermore, a statistical t-test will be conducted to rigorously compare the correlation results derived from both MATLAB and Python implementations, ensuring the validity and reliability of the adaptation process.

## Key Objectives
1. Code Adaptation: Convert the "pointpca2" project code from MATLAB to Python, ensuring that the core functionality and output remain consistent across both languages.
2. Feature Generation: Generate PCA features using both the original MATLAB code and the newly developed Python code. We will validate the equivalence of these features through statistical methods to ensure that both implementations produce comparable results.
3. Regression Analysis: Use the generated features to perform regression analysis against subjective scores in the dataset. This will involve using a set of regression models and fitting them with the features to predict the subjective scores. Calculate both Pearson and Spearman correlation coefficients for each regressor.
4. Correlation Analysis: Plot the correlation coefficients for each regressor. This analysis will provide insights into the linear and rank-order relationships between predicted scores and actual subjective scores.
5. Statistical Comparison: Conduct a t-test to statistically compare the correlation coefficients obtained from MATLAB and Python implementations. This step is critical to assess whether the differences in correlations (if any) are statistically significant, providing a quantitative measure of the adaptation's fidelity.

## Expected Outcomes
- A fully functional Python version of the "pointpca2" MATLAB project, verified for accuracy and equivalence.
- A detailed comparison of regression model performances using PCA features from both MATLAB and Python implementations.
- A statistical analysis report providing evidence on the equivalence (or differences) in correlation coefficients derived from both languages' codes.

## Prerequisites
- MATLAB (version R2023a tested)
- MATLAB Engine for Python (https://pypi.org/project/matlabengine/)
- anaconda3 (https://www.anaconda.com/)

## Installing
```bash
# Clone and cd into the repository
git clone https://github.com/akaTsunemori/pointpca2-python.git
cd pointpca2-python

# Setup the conda environment
conda env update --file environment.yml

# Activate the new env
conda activate pointpca2-python
```

## Usage
- #### pointpca2.py
    This is the project's main module. It replicates all the functions present in pointpca2's original code. The main function, lc_pointpca, should be called with the path for the reference and the path for the test point clouds, it returns an array consisting of the generated features.

- #### build_tables.py
    Builds the tables with pointpca2 features for both the Python and MATLAB algorithms.
    This script expects that the dataset's csv's columns follow the format:
  
    |SIGNAL  |REF     |SCORE   |LOCATION|REFLOCATION|ATTACK  |CLASS   |
    |--------|--------|--------|--------|-----------|--------|--------|

    It also expects that the informed locations are correct.

    Any exceptions happened during the computations of the lc_pointpca will be ignored
    and the row will be skipped.

    Checkpoints will be saved on ./tables/dataset_name, DO NOT remove, rename or change
    any of these files unless you've finished building the tables for the whole dataset.
    Doing so would compromise the (very simple) checkpoint system.

    It is expected that the setup for
    MATLAB (https://www.mathworks.com/products/matlab.html)
    and matlab.engine (https://pypi.org/project/matlabengine/)
    was properly done and tested.

    Tables for multiple datasets can be built at the same time, as long as you
    call this script multiple times with different values for dataset_name.

    Arguments for calling this from CLI are:
    - dataset_name: the name of the dataset. Example: **APSIPA**.
    - dataset_csv: path for the csv that describes the dataset. Example: **/home/user/Documents/APSIPA/apsipa.csv**. 
    - pointpca2_path: path for the original pointpca2 MATLAB code/repository. Example: **/home/user/Documents/pointpca2/**.
 
    The output will be saved in a new folder named "results", the checkpoints will be saved ona new folder named "tables".

- #### regressions.py
    This module uses the tables generated by *build_tables.py* and makes regressions using all available models from LazyPredict.
    The training/testing for these regressors will use two techniques: Leave One Group Out and Group K-Fold.

    Arguments for calling this from CLI are:
    - csv_path_MATLAB: path to the csv table corresponding to the features generated from the MATLAB version of the code, generated by build_tables.py. Example: **./results/APSIPA_pointpca2_MATLAB_cleaned.csv**.
    - csv_path_Python: path to the csv table corresponding to the features generated from the Python version of the code, generated by build_tables.py. Example: **./results/APSIPA_pointpca2_Python_cleaned.csv**.
    - dataset_name: the name of the dataset. Example: **APSIPA**.
 
    The output will be saved in a new folder named "regressions".

- #### plots.py
    This module uses the tables generated by *regressions.py* and plots the Pearson and Spearman correlation coefficients for each regressor.

    Arguments for calling this from CLI are:
    - csv_path_regression_MATLAB_LeaveOneGroupOut: path to the csv table corresponding to the Leave One Group Out regressions generated from the MATLAB version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_MATLAB_regression_LeaveOneGroupOut.csv**.
    - csv_path_regression_MATLAB_GroupKFold: path to the csv table corresponding to the Group K-Fold regressions generated from the MATLAB version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_MATLAB_regression_GroupKFold.csv**.
    - csv_path_regression_Python_LeaveOneGroupOut: path to the csv table corresponding to the Leave One Group Out regressions generated from the Python version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_Python_regression_LeaveOneGroupOut.csv**.
    - csv_path_regression_Python_GroupKFold: path to the csv table corresponding to the Group K-Fold regressions generated from the Python version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_Python_regression_GroupKFold.csv**.
    - dataset_name: the name of the dataset. Example: **APSIPA**.
 
    The output will be saved in a new folder named "plots".
  
- #### ttests.py
    This module uses the tables generated by *regressions.py* and conducts a t-test in order to statistically compare the correlation coefficients from the regression from the tables corresponding to the results of the MATLAB and Python codes, in order to provide a quantitative measure of the adaptation's fidelity.

    Arguments for calling this from CLI are:
    - csv_path_regression_MATLAB_LeaveOneGroupOut: path to the csv table corresponding to the Leave One Group Out regressions generated from the MATLAB version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_MATLAB_regression_LeaveOneGroupOut.csv**.
    - csv_path_regression_MATLAB_GroupKFold: path to the csv table corresponding to the Group K-Fold regressions generated from the MATLAB version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_MATLAB_regression_GroupKFold.csv**.
    - csv_path_regression_Python_LeaveOneGroupOut: path to the csv table corresponding to the Leave One Group Out regressions generated from the Python version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_Python_regression_LeaveOneGroupOut.csv**.
    - csv_path_regression_Python_GroupKFold: path to the csv table corresponding to the Group K-Fold regressions generated from the Python version of pointpca2, generated by regressions.py. Example: **./regressions/APSIPA_Python_regression_GroupKFold.csv**.
    - dataset_name: the name of the dataset. Example: **APSIPA**.
 
    The output will be saved in a new folder named "ttests".

## Results
Results for a number of datasets are stored in the [results](results) folder. Each folder contains the final checkpoints, cleaned feature tables, plots, regressions, and t-tests. Below are the results for the APSIPA dataset, which can also be found [here](results/APSIPA).

- #### Leave One Group Out
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

- #### Group K-Fold
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

*p-values rounded to 2 decimal places for better visibility*

## Acknowledgments
- [pointpca2](https://github.com/cwi-dis/pointpca2/)

## License
Licensed under the BSD 3-Clause Clear License

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)
