import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
#%config IPCompleter.greedy=True

df=pd.read_csv("insurance.csv")
df.head()

df.info()

df.describe()

sns.pairplot(df)

sns.heatmap(df.corr(), annot = True ,fmt=".2f")
plt.show()
#same as corr.

df.corr('spearman')

sns.displot(df.age)
sns.distplot(df.age)
sns.distplot(df.bmi)
sns.distplot(df.expenses)

sns.scatterplot(x='age',y='expenses',data=df,hue='smoker')
sns.scatterplot(x='bmi',y='expenses',data=df,hue='smoker')

df.sex.unique()
X=df[['age','sex','bmi','children','smoker']]
y=df.expenses


df.columns



df.sex.nunique()
#One hot encoding
    
cat_col=['smoker','region','sex']
num_col=[i for i in df.columns if i not in cat_col]
num_col

one_hot=pd.get_dummies(df[cat_col])
df=pd.concat([df[num_col],one_hot],axis=1)
df.head(10)

X = df.drop(['expenses'], axis = 1)
y = df.expenses

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LinearRegression  # Import Linear Regression model\n",
    
multiple_linear_reg = LinearRegression(fit_intercept=False)  # Create a instance for Linear Regression model\n",
multiple_linear_reg.fit(X_train, y_train)  # Fit data to the model"
   
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept= True)
regressor.fit(X_train, y_train) 

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df

regressor.intercept_

#just check with mean of the expenses
mean_exp=np.array(df.expenses)
np.mean(mean_exp,axis=0)

y_pred = regressor.predict(X_train)
df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
df

#Validating OLS Assumptions\n",
plt.scatter(y_pred, (y_train-y_pred))
plt.xlabel("Fitted values")
plt.ylabel("Residuals")

y_train-y_pred.mean()

sns.distplot(y_train-y_pred ,fit=None);
plt.xlabel('Residuals')

from scipy import stats
stats.probplot(y_train-y_pred, plot=plt)
plt.show()

import statsmodels.api as sm
X_endog = sm.add_constant(X_train)
X_endog1 = sm.add_constant(X_test)

res = sm.OLS(y_train, X_endog)
res.fit()

res.fit().summary()

y_pred1 = regressor.predict(X_test)
    
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
   
def mean_absolute_error(y_true,y_predict):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100  

y_pred1 = regressor.predict(X_train)
    
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred1)))


"# Importing evaluation metrics\n",
    
from sklearn.model_selection import cross_val_predict  # For K-Fold Cross Validation\n",
from sklearn.metrics import r2_score  # For find accuracy with R2 Score\n",
from sklearn.metrics import mean_squared_error  # For MSE\n",
from math import sqrt  # For squareroot operation"
  
## Accuracy with vanilla multiple linear regression\n",
    
# Prediction with training dataset:\n",
y_pred_MLR_train = multiple_linear_reg.predict(X_train)
    
# Prediction with testing dataset:\n",
y_pred_MLR_test = multiple_linear_reg.predict(X_test)
    
# Find training accuracy for this model:\n",
accuracy_MLR_train = r2_score(y_train, y_pred_MLR_train)
print("Training Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_train)
    
# Find testing accuracy for this model:\n",
accuracy_MLR_test = r2_score(y_test, y_pred_MLR_test)
print("Testing Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_test)
    
# Find RMSE for training data:\n",
RMSE_MLR_train = sqrt(mean_squared_error(y_train, y_pred_MLR_train))
print("RMSE for Training Data: ", RMSE_MLR_train)
   
# Find RMSE for testing data:
RMSE_MLR_test = sqrt(mean_squared_error(y_test, y_pred_MLR_test))
print("RMSE for Testing Data: ", RMSE_MLR_test)
    
# Prediction with 10-Fold Cross Validation:
y_pred_cv_MLR = cross_val_predict(multiple_linear_reg, X, y, cv=10)
    
# Find accuracy after 10-Fold Cross Validation
accuracy_cv_MLR = r2_score(y, y_pred_cv_MLR)
print("Accuracy for 10-Fold Cross Predicted Multiple Linear Regression Model: ", accuracy_cv_MLR)

from sklearn.preprocessing import PolynomialFeatures
polynomial_features = PolynomialFeatures(degree=2)  # Create a PolynomialFeatures instance in degree 3\n"
x_train_poly = polynomial_features.fit_transform(X_train)  # Fit and transform the training data to polynomial\n"
x_test_poly = polynomial_features.fit_transform(X_test)  # Fit and transform the testing data to polynomial\n",
    
polynomial_reg = LinearRegression(fit_intercept=False)  # Create a instance for Linear Regression model\n",
polynomial_reg.fit(x_train_poly, y_train)  # Fit data to the model"
   
   
x_train_poly.shape
x_test_poly.shape   

## Checking accuracy with Linear model with polynomial features\n",
    
# Prediction with training dataset:\n",
y_pred_PR_train = polynomial_reg.predict(x_train_poly)
  
# Prediction with testing dataset:\n",
y_pred_PR_test = polynomial_reg.predict(x_test_poly)
    
# Find training accuracy for this model:\n",
accuracy_PR_train = r2_score(y_train, y_pred_PR_train)
print("Training Accuracy for Polynomial Regression Model: ", accuracy_PR_train)
    
# Find testing accuracy for this model:\n",
accuracy_PR_test = r2_score(y_test, y_pred_PR_test)
print("Testing Accuracy for Polynomial Regression Model: ", accuracy_PR_test)
    
# Find RMSE for training data:
RMSE_PR_train = sqrt(mean_squared_error(y_train, y_pred_PR_train))
print("RMSE for Training Data: ", RMSE_PR_train)
    
# Find RMSE for testing data:\n",
RMSE_PR_test = sqrt(mean_squared_error(y_test, y_pred_PR_test))
print("RMSE for Testing Data: ", RMSE_PR_test)
   
# Prediction with 10-Fold Cross Validation:\n",
y_pred_cv_PR = cross_val_predict(polynomial_reg, polynomial_features.fit_transform(X), y, cv=10)
    
# Find accuracy after 10-Fold Cross Validation\n",
accuracy_cv_PR = r2_score(y, y_pred_cv_PR)
print("Accuracy for 10-Fold Cross Predicted Polynomial Regression Model: ", accuracy_cv_PR)

###decision tree 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


    
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
    
#%matplotlib inline
sns.set_style("whitegrid")
#plt.style.use("fivethirtyeight")

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(X_train, y_train)

decision_tree_reg.predict(X_test)

df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':decision_tree_reg.predict(X_test)})
df

## Checking accuracy with Decision trees \n",
    
# Prediction with training dataset:\n",
y_pred_DTR_train = decision_tree_reg.predict(X_train)
    
# Prediction with testing dataset:\n",
y_pred_DTR_test = decision_tree_reg.predict(X_test)
    
# Find training accuracy for this model:\n",
accuracy_DTR_train = r2_score(y_train, y_pred_DTR_train)
print("Training Accuracy for Decision Tree Regression Model: ", accuracy_DTR_train)
    
# Find testing accuracy for this model:\n",
accuracy_DTR_test = r2_score(y_test, y_pred_DTR_test)
print("Testing Accuracy for Decision Tree Regression Model: ", accuracy_DTR_test)
    
# Find RMSE for training data:\n",
RMSE_DTR_train = sqrt(mean_squared_error(y_train, y_pred_DTR_train))
print("RMSE for Training Data: ", RMSE_DTR_train)
    
# Find RMSE for testing data:\n",
RMSE_DTR_test = sqrt(mean_squared_error(y_test, y_pred_DTR_test))
print("RMSE for Testing Data: ", RMSE_DTR_test)
    
# Prediction with 10-Fold Cross Validation:\n",
y_pred_cv_DTR = cross_val_predict(decision_tree_reg, X, y, cv=10)
   
# Find accuracy after 10-Fold Cross Validation\n",
accuracy_cv_DTR = r2_score(y, y_pred_cv_DTR)
print("Accuracy for 10-Fold Cross Predicted Decision Tree Regression Model: ", accuracy_cv_DTR)
  
#we have not used random oversampling, undersampling and smote as the
#  dataset is not imbalanced. the mentioned techinques can be used for both regression as well as classification

## Hyperparameter tuning in Decision trees\n",
   
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
    
params = { 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
    }
   
tree_clf = DecisionTreeRegressor(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, n_jobs=-1, verbose=1, cv=3)
tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params}")
    
tree_clf = DecisionTreeRegressor(**best_params)
tree_clf.fit(X_train, y_train)
   
"# Prediction with training dataset:\n",
y_pred_DTR_train = tree_clf.predict(X_train)
    
# Prediction with testing dataset:\n",
y_pred_DTR_test = tree_clf.predict(X_test)
    
# Find RMSE for training data:\n",
RMSE_DTR_train = sqrt(mean_squared_error(y_train, y_pred_DTR_train))
print("RMSE for Training Data: ", RMSE_DTR_train)
    
# Find RMSE for testing data:\n",
RMSE_DTR_test = sqrt(mean_squared_error(y_test, y_pred_DTR_test))
print("RMSE for Testing Data: ", RMSE_DTR_test)
    
# Find training accuracy for this model:\n",
accuracy_DTR_train = r2_score(y_train, y_pred_DTR_train)
print("Training Accuracy for Decision Tree Regression Model: ", accuracy_DTR_train)
    
# Find testing accuracy for this model:",
accuracy_DTR_test = r2_score(y_test, y_pred_DTR_test)
print("Testing Accuracy for Decision Tree Regression Model: ", accuracy_DTR_test)
    
# Prediction with 10-Fold Cross Validation:\n",
y_pred_cv_DTR = cross_val_predict(tree_clf, X, y, cv=10)
    
# Find accuracy after 10-Fold Cross Validation\n",
accuracy_cv_DTR = r2_score(y, y_pred_cv_DTR)
print("Accuracy for 10-Fold Cross Predicted Decision Tree Regression Model: ", accuracy_cv_DTR)


from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regression model\n",
    
random_forest_reg = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=13)  # Create a instance for Random Forest Regression model\n",
random_forest_reg.fit(X_train, y_train)  # Fit data to the model"

"# Prediction with training dataset:\n",
y_pred_RF_train = random_forest_reg.predict(X_train)

# Prediction with testing dataset:\n",
y_pred_RF_test = random_forest_reg.predict(X_test)

# Find RMSE for training data:\n",
RMSE_RF_train = sqrt(mean_squared_error(y_train, y_pred_RF_train))
print("RMSE for Training Data: ", RMSE_RF_train)

# Find RMSE for testing data:\n",
RMSE_RF_test = sqrt(mean_squared_error(y_test, y_pred_RF_test))
print("RMSE for Testing Data: ", RMSE_RF_test)

# Find training accuracy for this model:\n",
accuracy_RF_train = r2_score(y_train, y_pred_RF_train)
print("Training Accuracy for Decision Tree Regression Model: ", accuracy_RF_train)
  
# Find testing accuracy for this model:",
accuracy_RF_test = r2_score(y_test, y_pred_RF_test)
print("Testing Accuracy for Decision Tree Regression Model: ", accuracy_RF_test)
      
# Prediction with 10-Fold Cross Validation:\n",
y_pred_cv_RF = cross_val_predict(random_forest_reg, X, y, cv=10)
      
# Find accuracy after 10-Fold Cross Validation\n",
# Find accuracy after 10-Fold Cross Validation\n",
accuracy_cv_RF = r2_score(y, y_pred_cv_RF)
print("Accuracy for 10-Fold Cross Predicted Decision Tree Regression Model: ", accuracy_cv_RF)
     

## Hyperparameter tuning in Random forests\n",
   
n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2,3,4,5,6]
max_depth.append(None)
#min_samples_split = [2, 5, 10]\n",
#min_samples_leaf = [1, 2, 4, 10]\n",
    
params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
                   'max_depth': max_depth} 
#'min_samples_split': min_samples_split,             
#'min_samples_leaf': min_samples_leaf}\n",
    
rf_clf = RandomForestRegressor(random_state=42)


   
rf_cv = GridSearchCV(rf_clf, params_grid, cv=3, verbose=2,n_jobs = -1)
    
rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")
    
rf_clf = RandomForestRegressor(**best_params)
rf_clf.fit(X_train, y_train)

## Checking accuracy with Tuned RFs\n",
accuracy_RFR_train=r2_score(y_train, y_pred_RF_train)
accuracy_RFR_test=r2_score(y_test, y_pred_RF_test)
# Prediction with training dataset:\n",
y_pred_RFR_train = rf_clf.predict(X_train)

# Prediction with testing dataset:\n",
y_pred_RFR_test = rf_clf.predict(X_test)
    
# Find RMSE for training data:\n",
RMSE_RFR_train = sqrt(mean_squared_error(y_train, y_pred_RFR_train))
print("RMSE for Training Data: ", RMSE_RFR_train)
   
# Find RMSE for testing data:\n",
RMSE_RFR_test = sqrt(mean_squared_error(y_test, y_pred_RFR_test))
print("RMSE for Testing Data: ", RMSE_RFR_test)


# Import SVR model\n",
from sklearn.svm import SVR  
support_vector_reg = SVR(gamma="auto", kernel="rbf", C=1000 )  
# Create a instance for Support Vector Regression model\n",
support_vector_reg.fit(X_train, y_train)  # Fit data to the model"

"# Prediction with training dataset:\n",
y_pred_svr_train = support_vector_reg.predict(X_train)

# Prediction with testing dataset:\n",
y_pred_svr_test = support_vector_reg.predict(X_test)

# Find RMSE for training data:\n",
RMSE_svr_train = sqrt(mean_squared_error(y_train, y_pred_svr_train))
print("RMSE for Training Data: ", RMSE_svr_train)

# Find RMSE for testing data:\n",
RMSE_svr_test = sqrt(mean_squared_error(y_test, y_pred_svr_test))
print("RMSE for Testing Data: ", RMSE_svr_test)

# Find training accuracy for this model:\n",
accuracy_svr_train = r2_score(y_train, y_pred_svr_train)
accuracy_svr_test = r2_score(y_test, y_pred_svr_test)
print("Training Accuracy for svRegression Model: ", accuracy_svr_train)

# Find testing accuracy for this model:",
accuracy_svr_test = r2_score(y_test, y_pred_svr_test)
print("Testing Accuracy for svr Model: ", accuracy_svr_test)
  
# Prediction with 10-Fold Cross Validation:\n",
y_pred_cv_svr = cross_val_predict(support_vector_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation\n",
accuracy_cv_svr = r2_score(y, y_pred_cv_svr)
print("Accuracy for 10-Fold Cross Predicted Decision Tree Regression Model: ", accuracy_cv_svr)
     
## Hyper parameter tuning the SVM\n",
   
from sklearn.model_selection import GridSearchCV
   
param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100,1000,10000,100000], 
                  'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001], 
                 'kernel': ['rbf', 'linear']} 
    
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=1, cv=5,n_jobs = -1)
grid.fit(X_train, y_train)
    
best_params = grid.best_params_
print(f"Best params: {best_params}")
    
svm_clf = SVR(**best_params)
svm_clf.fit(X_train, y_train)
  
## Checking accuracy with Tuned SVMs\n",
accuracy_SVR_train=r2_score(y_train, y_pred_svr_train)   
accuracy_SVR_test=r2_score(y_test, y_pred_svr_test) 
# Prediction with training dataset:\n",
y_pred_SVR_train = svm_clf.predict(X_train)
    
# Prediction with testing dataset:\n",
y_pred_SVR_test = svm_clf.predict(X_test)
    
# Find RMSE for training data:\n",
RMSE_SVR_train = sqrt(mean_squared_error(y_train, y_pred_SVR_train))
print("RMSE for Training Data: ", RMSE_SVR_train)
    
# Find RMSE for testing data:",
RMSE_SVR_test = sqrt(mean_squared_error(y_test, y_pred_SVR_test))
print("RMSE for Testing Data: ", RMSE_SVR_test)
 

 #check the changed namess 
# Compare all results in one table\n",
training_accuracies = [accuracy_MLR_train, accuracy_PR_train, accuracy_DTR_train, accuracy_RFR_train, accuracy_SVR_train]
testing_accuracies = [accuracy_MLR_test, accuracy_PR_test, accuracy_DTR_test, accuracy_RFR_test, accuracy_SVR_test]
training_RMSE = [RMSE_MLR_train, RMSE_PR_train, RMSE_DTR_train, RMSE_RFR_train, RMSE_SVR_train]
testing_RMSE = [RMSE_MLR_test, RMSE_PR_test, RMSE_DTR_test, RMSE_RFR_test, RMSE_SVR_test]
#cv_accuracies = [accuracy_cv_MLR, accuracy_cv_PR, accuracy_cv_DTR, accuracy_cv_RFR, "NA"]
    
parameters = ["fit_intercept=False", "fit_intercept=False", "max_depth=4", "n_estimators=500, max_depth=4", "kernel=”rbf”, C=1000"]
   
table_data = {"Parameters": parameters, "Training Accuracy": training_accuracies, "Testing Accuracy": testing_accuracies, 
                  "Training RMSE": training_RMSE, "Testing RMSE": testing_RMSE}# "10-Fold Score": cv_accuracies}
model_names = ["Multiple Linear Regression", "Polynomial Regression", "Decision Tree Regression", "Random Forest Regression", "Support Vector Regression"]
    
table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe 

import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

xgb_model = XGBRegressor(
        objective = 'reg:squarederror',
        colsample_bytree = 0.7,
        learning_rate = 0.01,
        max_depth = 3,
        min_child_weight = 5,
        n_estimators = 500,
        subsample = 0.7)

xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb1 = xgb_model.predict(X_train)


rmse_xgb = sqrt(mean_squared_error(y_test, y_pred_xgb))
rmse_xgb1 = sqrt(mean_squared_error(y_train, y_pred_xgb1))


print("RMSE Training data:", rmse_xgb1)
print("RMSE Testing data:", rmse_xgb)

def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1,0.3],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           verbose = 1,
                           n_jobs = -1
                          )
    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

from math import sqrt
hyperParameterTuning(X_train, y_train)

