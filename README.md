# Causality in Data Cleaning
Project for "Machine Learning for Data Cleaning", Winter 2019, University of Waterloo

In this project, we use methods of statistical causal inference to perform hypothesis tests of assumed causes for data errors.

## Project Structure
We use existing data sets as linked in the Python notebooks. 
Given the data set, we reformat it such that it can be loaded in Data X-Ray. For this, use the `create-*.ipynb` notebooks.

Once we have an assumed cause, we perform the actual causal analysis. These analyses can be found in the `causal-analysis-*.ipynb` notebooks.

Commonly used code for propensity score stratification is in `utils/treatment_effect.py`, code to read X-Ray output in `utils/xray_util.py`
