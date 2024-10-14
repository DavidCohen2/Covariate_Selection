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

Covariate_Selection/
│
├── ate_estimate.py                             # Script for ATE estimation
├── ATE_folds.py                                # Script handling ATE with multiple folds
├── ATE_folds_simple.py                         # A simplified version of ATE folds calculation
├── Covariate Selection2.py                     # Main script for covariate selection
├── data_generating_process.py                  # Script to generate synthetic data
├── main.py                                     # Entry point for running the project
├── plot_propensity_score_distribution.py       # Script to plot the distribution of propensity scores
├── README.md                                   # Readme file explaining the project
├── requirements_python_3_08.txt                # File listing project dependencies
└── utilities.py                                # Utility functions used across the project

## How to use SDM and SSDM
