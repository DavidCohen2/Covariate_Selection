# ATE Covariate Selection

In this project, we tackled the challenge of accurately estimating the ATE while selecting covariates that enhance ATE 
assessment. We underscored the significance of the overlap and ignorability assumptions, demonstrating how these two 
assumptions can conflict. Specifically, including a confounder may disrupt the overlap condition, while excluding it could 
violate the ignorability assumption. To address this issue, we proposed a method based on Cohen's d metric to 
determine whether a confounder should be included. Through multiple simulations, we illustrated our findings and 
highlighted the potential of our method for effective covariate selection.

*David Cohen*, *Harel Mendelman*

## Installing
You can install all the required packages by executing the following command:

```bash
pip install -r requirements_python_3_08.txt
```

## Project files
```bash
Covariate_Selection/
├── ate_estimate.py                        # Script for ATE estimation
├── ATE_folds.py                           # Script for simulation 1
├── ATE_folds_simple.py                    # Script for simulation 2
├── data_generating_process.py             # Script to generate synthetic data
├── toy_problem.py                                # Script for toy problem simulation
├── plot_propensity_score_distribution.py  # Script to plot the distribution of propensity scores
├── README.md                              # Readme file explaining the project
├── requirements_python_3_08.txt           # File listing project dependencies
└── utilities.py                           # Utility functions used across the project
```