import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,explained_variance_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import BayesianRidge,ElasticNet,HuberRegressor,Lasso,LinearRegression,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
all_sensors_df = pd.read_csv('/updated_csv_files/IP3_Data.csv',usecols=['AccelerationX','AccelerationY','AccelerationZ','MagneticFieldX','MagneticFieldY','MagneticFieldZ','Z-AxisAgle(Azimuth)','X-AxisAngle(Pitch)','Y-AxisAngle(Roll)','tag','RSSI_2','RSSI_3','RSSI_4','RSSI_5','RSSI_6','RSSI_7','RSSI_8','RSSI_9','RSSI_10'])
ss_enc = StandardScaler()
mm_enc = MinMaxScaler()
all_sensors_df[['AccelerationX','AccelerationY','AccelerationZ','X-AxisAngle(Pitch)','Y-AxisAngle(Roll)']] = ss_enc.fit_transform(all_sensors_df[['AccelerationX','AccelerationY','AccelerationZ','X-AxisAngle(Pitch)','Y-AxisAngle(Roll)']])
all_sensors_df[['MagneticFieldX','MagneticFieldY','MagneticFieldZ','Z-AxisAgle(Azimuth)']] = mm_enc.fit_transform(all_sensors_df[['MagneticFieldX','MagneticFieldY','MagneticFieldZ','Z-AxisAgle(Azimuth)']])

def get_train_test_samples(id1:int):
    X = all_sensors_df.drop(columns=['RSSI_2','RSSI_3','RSSI_4','RSSI_5','RSSI_6','RSSI_7','RSSI_8','RSSI_9','RSSI_10'])
    y = all_sensors_df[['RSSI_'+str(id1)]]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values.reshape(-1)
    y_test = y_test.values.reshape(-1)
    return (X_train,X_test,y_train,y_test)

def training_classifier(x_train,y_train,x_test,y_test):
    classifiers = {
            "Linear Regression" : LinearRegression(),
            "Ridge Regression" : Ridge(random_state=1234),
            "Lasso Regression" : Lasso(random_state=1234),
            "Elastic Net Regression" : ElasticNet(random_state=1234),
            "Bayesian Ridge Regression" : BayesianRidge(),
            "Decision Tree Regression (DTR)" : DecisionTreeRegressor(random_state=1234),
            "Random Forest Regression" : RandomForestRegressor(random_state=1234),
            "Gradient Boosting Regression" : GradientBoostingRegressor(random_state=1234),
            "K-Nearest Neighbors (KNN) Regression" : KNeighborsRegressor(n_neighbors=5),
            "Huber Regressor" : HuberRegressor(max_iter=1000),
    }
    results = []
    for name,clf in classifiers.items():
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        mse_error = mean_squared_error(y_test,y_pred)
        rmse_error = np.sqrt(mse_error)
        true_minus_pred = y_test-y_pred
        result = {
            "Model_Name" : name,
            "Mean Squared Error" : mse_error,
            "Root Mean Squared Error" : rmse_error,
            "Normalized Root Mean Squared Error" : np.sqrt(np.mean((true_minus_pred)**2))/np.max(y_test)-np.min(y_test),
            "R2 Score" : r2_score(y_test,y_pred),
            "Mean Absolute Error" : mean_absolute_error(y_test,y_pred),
            "Mean Absolute Percentage Error" : np.mean(np.abs((true_minus_pred)/y_test))*100,
            "Mean Bias Diviation Error" : np.mean(true_minus_pred),
            "Mean Absolute Scaled Error" : np.mean(np.abs(true_minus_pred))/np.mean(np.abs(y_test[1:]-y_test[:-1])),
            "Explained Variance Score" : explained_variance_score(y_test,y_pred),
        }
        results.append(result)

    result_df = pd.DataFrame(data=results)
    return result_df

def get_custom_random_grid_search(x_train,y_train,x_test,y_test,n_iter,cv):
    classifiers = {
            "Ridge Regression" : Ridge(random_state=1234),
            "Lasso Regression" : Lasso(random_state=1234),
            "Elastic Net Regression" : ElasticNet(random_state=1234),
            "Bayesian Ridge Regression" : BayesianRidge(),
            "Decision Tree Regression (DTR)" : DecisionTreeRegressor(random_state=1234),
            "Random Forest Regression" : RandomForestRegressor(random_state=1234),
            "Gradient Boosting Regression" : GradientBoostingRegressor(random_state=1234),
            "K-Nearest Neighbors (KNN) Regression" : KNeighborsRegressor(n_neighbors=5),
            "Huber Regressor" : HuberRegressor(max_iter=1000),
    }
    models_and_params = [
        (
            "Ridge Regression",
            {
                'alpha' : [0.5,0.7,1,1.47,2.16,3.19,4.67,6.87],
                'solver' : ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'],
                'tol' : [1e-4,1e-5,1e-6,1e-7],
                }
            ),
        (
            "Lasso Regression",
            {
                'alpha' : [0.5,0.7,1,1.47,2.16,3.19,4.67,6.87],
                'selection' : ['cyclic','random'],
                'tol' : [1e-4,1e-5,1e-6,1e-7],
                }
            ),
        (
            "Elastic Net Regression",
            {
                'alpha' : [0.5,0.7,1,1.47,2.16,3.19,4.67,6.87],
                'l1_ratio' : [0.25,0.5,0.75,1],
                'selection' : ['cyclic','random'],
                'tol' : [1e-4,1e-5,1e-6,1e-7],
                }
            ),
        (
            "Bayesian Ridge Regression",
            {
                'tol' : [1e-4,1e-5,1e-6,1e-7],
                'alpha_1' : [1e-4,1e-5,1e-6,1e-7],
                'alpha_2' : [1e-4,1e-5,1e-6,1e-7],
                'lambda_1' : [1e-4,1e-5,1e-6,1e-7],
                'lambda_2' : [1e-4,1e-5,1e-6,1e-7],
                }
            ),
        (
            "Decision Tree Regression (DTR)",
            {
                'criterion' : ['squared_error','friedman_mse','absolute_error'],
                'splitter' : ['best','random'],
                'max_features' : ['sqrt','log2'],
                }
            ),
        (
            "Random Forest Regression",
            {
                'n_estimators' : [50,100,150,200],
                'max_features' : ['sqrt','log2'],
                }
            ),
        (
            "Gradient Boosting Regression",
            {
                'loss' : ['squared_error','absolute_error','huber','quantile'],
                'learning_rate' : [1e-1,1e-2,1e-3,1e-4],
                'n_estimators' : [50,100,150,200],
                'criterion' : ['squared_error','friedman_mse'],
                'tol' : [1e-4,1e-5,1e-6,1e-7],
                }
            ),
        (
            "K-Nearest Neighbors (KNN) Regression",
            {
                'n_neighbors' : [3,4,5,6],
                'weights' : ['uniform','distance'],
                'algorithm' : ['auto','ball_tree','kd_tree','brute'],
                }
            ),
        (
            "Huber Regressor",
            {
                'epsilon' : [1,1.47,2.16,3.19,4.67,6.87],
                'tol' : [1e-4,1e-5,1e-6,1e-7],
                'alpha' : [1e-4,1e-5,1e-6,1e-7],
                }
            ),
    ]

    results = []
    prediction_results = []

    for model_name,param_dist in models_and_params:
        clf = RandomizedSearchCV(classifiers[model_name],param_distributions=param_dist,n_iter=n_iter,cv=cv,random_state=1234,n_jobs=-1)
        clf.fit(x_train,y_train)
        result = {
                    'Model_Name' : model_name,
                    'best_params': clf.best_params_,
                    'best_score': clf.best_score_,
                    }
        results.append(result)
    clf.set_params(**clf.best_params_)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    mse_error = mean_squared_error(y_test,y_pred)
    rmse_error = np.sqrt(mse_error)
    true_minus_pred = y_test-y_pred
    result = {
        "Model_Name" : model_name,
        "Mean Squared Error" : mse_error,
        "Root Mean Squared Error" : rmse_error,
        "Normalized Root Mean Squared Error" : np.sqrt(np.mean((true_minus_pred)**2))/np.max(y_test)-np.min(y_test),
        "R2 Score" : r2_score(y_test,y_pred),
        "Mean Absolute Error" : mean_absolute_error(y_test,y_pred),
        "Mean Absolute Percentage Error" : np.mean(np.abs((true_minus_pred)/y_test))*100,
        "Mean Bias Deviation Error" : np.mean(true_minus_pred),
        "Mean Absolute Scaled Error" : np.mean(np.abs(true_minus_pred))/np.mean(np.abs(y_test[1:]-y_test[:-1])),
        "Explained Variance Score" : explained_variance_score(y_test,y_pred),
    }
    prediction_results.append(result)
    result_df = pd.DataFrame(data=results)
    prediction_results = pd.DataFrame(data=prediction_results)
    return result_df,prediction_results

vanila_predictions = pd.DataFrame()
optimized_predictions = pd.DataFrame()
best_params_report = pd.DataFrame()

for i in range(2,7):
    X_train,X_test,y_train,y_test = get_train_test_samples(id1=i)
    prediction_results_rssi = training_classifier(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test)
    prediction_results_rssi['Predictor'] = 'RSSI_'+str(i)
    vanila_predictions = pd.concat(objs=[vanila_predictions,prediction_results_rssi])
    grid_search_results,new_predictions = get_custom_random_grid_search(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,n_iter=10,cv=7)
    grid_search_results['Predictor'] = 'RSSI_'+str(i)
    optimized_predictions = pd.concat(objs=[optimized_predictions,new_predictions])
    best_params_report = pd.concat(objs=[best_params_report,grid_search_results])

vanila_predictions.columns = ['Un_Optimized_'+str(x) for x in vanila_predictions.columns]
optimized_predictions.columns = ['Optimized_'+str(x) for x in optimized_predictions.columns]
optimized_predictions = pd.concat(objs=[optimized_predictions,vanila_predictions],axis=1)
optimized_predictions = optimized_predictions.drop(columns=['Optimized_Model_Name','Optimized_Predictor'])
for c_name in ['R2 Score','Explained Variance Score',]:
    optimized_predictions['%Change_'+str(c_name)] = (((optimized_predictions['Optimized_'+str(c_name)] - optimized_predictions['Un_Optimized_'+str(c_name)]) / optimized_predictions['Un_Optimized_'+str(c_name)])*100)
for c_name in ['Mean Squared Error','Root Mean Squared Error','Normalized Root Mean Squared Error','Mean Absolute Error','Mean Absolute Percentage Error','Mean Bias Diviation Error','Mean Absolute Scaled Error',]:
    optimized_predictions['%Change_'+str(c_name)] = (((optimized_predictions['Un_Optimized_'+str(c_name)] - optimized_predictions['Optimized_'+str(c_name)]) / optimized_predictions['Un_Optimized_'+str(c_name)])*100)

vanila_predictions.to_csv('/updated_csv_files/vanila_predictions.csv',index=False)
optimized_predictions.to_csv('/updated_csv_files/optimized_predictions.csv',index=False)
best_params_report.to_csv('/updated_csv_files/best_params_report.csv',index=False)


