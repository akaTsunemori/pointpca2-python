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
    SIGNAL,REF,SCORE,LOCATION,REFLOCATION,ATTACK,CLASS
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
Results for a number of datasets are stored in the [results](results) folder. Each folder contains the final checkpoints, cleaned feature tables, plots, regressions, and t-tests. Below are the results for the APSIPA dataset.

- #### Leave One Group Out
![APSIPA_LeaveOneGroupOut](results/APSIPA/plots/APSIPA_LeaveOneGroupOut.png)
|Model                        |p-value (Pearson)  |p_value <= 0.05 (Pearson)|p-value (Spearman) |p_value <= 0.05 (Spearman)|
|-----------------------------|-------------------|-------------------------|-------------------|--------------------------|
|AdaBoostRegressor            |0.5940428018205545 |False                    |0.5808245915349846 |False                     |
|BaggingRegressor             |0.8652964659523231 |False                    |0.4816183488170528 |False                     |
|BayesianRidge                |0.8976407359452343 |False                    |0.9035159044010612 |False                     |
|DecisionTreeRegressor        |0.3662507721820587 |False                    |0.35245932841903327|False                     |
|ElasticNet                   |0.9957604703933023 |False                    |0.9899195928284114 |False                     |
|ElasticNetCV                 |0.8682635047675572 |False                    |0.9353712816730515 |False                     |
|ExtraTreeRegressor           |0.5863824828571975 |False                    |0.5965054844015201 |False                     |
|ExtraTreesRegressor          |0.9227503133542578 |False                    |0.9136274699718901 |False                     |
|GammaRegressor               |0.9959208784510398 |False                    |0.982605282923443  |False                     |
|GaussianProcessRegressor     |0.9972276634322356 |False                    |0.994677902526238  |False                     |
|GradientBoostingRegressor    |0.7791899833199756 |False                    |0.9993078999018824 |False                     |
|HistGradientBoostingRegressor|0.737029928490573  |False                    |0.7851947529735057 |False                     |
|HuberRegressor               |0.9202823448745467 |False                    |0.9976137035820045 |False                     |
|KNeighborsRegressor          |0.9973990658622369 |False                    |0.9874291054379765 |False                     |
|KernelRidge                  |0.8909245683847816 |False                    |0.976001593996167  |False                     |
|LGBMRegressor                |0.8823886047089274 |False                    |0.8843344075085421 |False                     |
|Lars                         |0.37578542783329993|False                    |0.32862193089585967|False                     |
|LarsCV                       |0.32135562234686876|False                    |0.3049973714025437 |False                     |
|Lasso                        |0.9996220782341332 |False                    |0.991305371424893  |False                     |
|LassoCV                      |0.9466313255220544 |False                    |0.949737917898182  |False                     |
|LassoLars                    |0.9996228156590281 |False                    |0.991305371424893  |False                     |
|LassoLarsCV                  |0.958655764360453  |False                    |0.9215198468541572 |False                     |
|LassoLarsIC                  |0.8595657014799178 |False                    |0.9184430050724466 |False                     |
|LinearRegression             |0.8946373842597292 |False                    |0.9551638705204899 |False                     |
|LinearSVR                    |0.8031173974022923 |False                    |0.9712985290740261 |False                     |
|MLPRegressor                 |0.993596524152209  |False                    |0.9220760696146375 |False                     |
|NuSVR                        |0.9898518386889595 |False                    |0.9819546176346696 |False                     |
|OrthogonalMatchingPursuit    |0.9923674999778437 |False                    |0.9829171245333703 |False                     |
|OrthogonalMatchingPursuitCV  |0.9991381468705023 |False                    |0.9749597060549478 |False                     |
|PassiveAggressiveRegressor   |0.9995410553178112 |False                    |0.9340553215756804 |False                     |
|PoissonRegressor             |0.9944431327265908 |False                    |0.9884562246809714 |False                     |
|RANSACRegressor              |0.3319648605597898 |False                    |0.5206709094982876 |False                     |
|RandomForestRegressor        |0.9847617300158265 |False                    |0.9007089123985869 |False                     |
|Ridge                        |0.8909245683859892 |False                    |0.976001593996167  |False                     |
|RidgeCV                      |0.960151417871339  |False                    |0.9414494941250067 |False                     |
|SGDRegressor                 |0.9983105123224045 |False                    |1.0                |False                     |
|SVR                          |0.9955856740922825 |False                    |0.9911473580005008 |False                     |
|TransformedTargetRegressor   |0.8946373842597292 |False                    |0.9551638705204899 |False                     |
|TweedieRegressor             |0.995630648997472  |False                    |0.9974167788781709 |False                     |
|XGBRegressor                 |0.9995757472734221 |False                    |0.8866083519208414 |False                     |

- #### Group K-Fold
![APSIPA_GroupKFold](results/APSIPA/plots/APSIPA_GroupKFold.png)
|Model                        |p-value (Pearson)  |p_value <= 0.05 (Pearson)|p-value (Spearman) |p_value <= 0.05 (Spearman)|
|-----------------------------|-------------------|-------------------------|-------------------|--------------------------|
|AdaBoostRegressor            |0.594042801820559  |False                    |0.5808245915349846 |False                     |
|BaggingRegressor             |0.8652964659523231 |False                    |0.4816183488170528 |False                     |
|BayesianRidge                |0.8976407359452372 |False                    |0.9035159044010612 |False                     |
|DecisionTreeRegressor        |0.3662507721820587 |False                    |0.35245932841903327|False                     |
|ElasticNet                   |0.9957604703933023 |False                    |0.9899195928284084 |False                     |
|ElasticNetCV                 |0.8682635047675572 |False                    |0.9353712816730548 |False                     |
|ExtraTreeRegressor           |0.5863824828571975 |False                    |0.5965054844015216 |False                     |
|ExtraTreesRegressor          |0.9227503133542578 |False                    |0.9136274699718901 |False                     |
|GammaRegressor               |0.9959208784510443 |False                    |0.982605282923443  |False                     |
|GaussianProcessRegressor     |0.9972276634322351 |False                    |0.994677902526238  |False                     |
|GradientBoostingRegressor    |0.7791899833199756 |False                    |0.999307899901877  |False                     |
|HistGradientBoostingRegressor|0.737029928490573  |False                    |0.7851947529735057 |False                     |
|HuberRegressor               |0.9202823448745512 |False                    |0.9976137035820068 |False                     |
|KNeighborsRegressor          |0.9973990658622334 |False                    |0.9874291054379765 |False                     |
|KernelRidge                  |0.8909245683847816 |False                    |0.9760015939961703 |False                     |
|LGBMRegressor                |0.8823886047089274 |False                    |0.8843344075085421 |False                     |
|Lars                         |0.3757854278332997 |False                    |0.32862193089585967|False                     |
|LarsCV                       |0.32135562234686876|False                    |0.3049973714025437 |False                     |
|Lasso                        |0.9996220782341346 |False                    |0.9913053714248918 |False                     |
|LassoCV                      |0.9466313255220493 |False                    |0.949737917898182  |False                     |
|LassoLars                    |0.9996228156590281 |False                    |0.9913053714248918 |False                     |
|LassoLarsCV                  |0.9586557643604479 |False                    |0.9215198468541609 |False                     |
|LassoLarsIC                  |0.8595657014799177 |False                    |0.9184430050724479 |False                     |
|LinearRegression             |0.8946373842597292 |False                    |0.9551638705204911 |False                     |
|LinearSVR                    |0.803117397402289  |False                    |0.9712985290740261 |False                     |
|MLPRegressor                 |0.993596524152209  |False                    |0.9220760696146375 |False                     |
|NuSVR                        |0.9898518386889595 |False                    |0.9819546176346696 |False                     |
|OrthogonalMatchingPursuit    |0.9923674999778437 |False                    |0.9829171245333703 |False                     |
|OrthogonalMatchingPursuitCV  |0.9991381468705023 |False                    |0.9749597060549511 |False                     |
|PassiveAggressiveRegressor   |0.9995410553178112 |False                    |0.9340553215756822 |False                     |
|PoissonRegressor             |0.9944431327265908 |False                    |0.9884562246809714 |False                     |
|RANSACRegressor              |0.3319648605597898 |False                    |0.5206709094982876 |False                     |
|RandomForestRegressor        |0.9847617300158322 |False                    |0.9007089123985869 |False                     |
|Ridge                        |0.8909245683859892 |False                    |0.9760015939961703 |False                     |
|RidgeCV                      |0.9601514178713362 |False                    |0.9414494941250067 |False                     |
|SGDRegressor                 |0.9983105123224001 |False                    |1.0                |False                     |
|SVR                          |0.9955856740922825 |False                    |0.9911473580005008 |False                     |
|TransformedTargetRegressor   |0.8946373842597292 |False                    |0.9551638705204911 |False                     |
|TweedieRegressor             |0.995630648997468  |False                    |0.9974167788781709 |False                     |
|XGBRegressor                 |0.9995757472734221 |False                    |0.8866083519208414 |False                     |

## Acknowledgments
- [pointpca2](https://github.com/cwi-dis/pointpca2/)

## License
Licensed under the BSD 3-Clause Clear License

---

> GitHub [@akaTsunemori](https://github.com/akaTsunemori)
