## CERN-clollision-data

Author: Mohammadreza Ebrahimi  
Email: [m.reza.ebrahimi1995@gmail.com](mailto:m.reza.ebrahimi1995@gmail.com)
***

This repository is included a perfect example for **Machin learning**.

It is about electron mass prediction by utilizing _CERN electron collision data_. 
We examined different model as follows

- LinearRegression
- DecisionTree
- RandomForest

Then, after training model by using them, we examined them
for data which prepared by **Polynomial Feature**.  
At the end, we would train several perfect models with low _cost function_ where the best **RMSE** would be gotten as **1.6** for the test set. 

Still this model can gives us the better **MSE** or **RMSE**
by tuning **Hyperparameter**.  
If you have any question or suggestion please [contact](mailto:m.reza.ebrahimi1995@gmail.com) me.   

### Summery of Results 

**Prediction of Mass**  
Mass of electron (min=2.00 - max=109.99)  
Cross Validation of Training dataset  
Shape of data : (100000, 19)

Model|RMSE
--------|------
Linear Regression | 19.29
LASSO (alpha=0.01)| 19.50
Decision Tree | 12.07
Random Forest|6.51
||
| |  ***Polynomial Features (degree=2)***
||
Linear Regression |4.57
Decision Tree|2.69
Random Forest| 1.68
||
| | ***Test dataset***
||
Random Forest | 1.60
