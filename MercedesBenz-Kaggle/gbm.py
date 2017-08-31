"""
Author - Raunak Mundada
Date - 6/17/2017
"""

"""
Runs a grid search for gradient boosting machines
on Mercedes-Benz dataset
"""

import os
os.chdir("D:/ML_Projects/MercedesBenz-Kaggle/")
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
import sklearn.metrics as mt
from sklearn import ensemble

np.random.seed(12548)

def read_data():
	train_data = pd.read_csv("./data/train.csv")
	test_data = pd.read_csv("./data/test.csv")

	return train_data, test_data


def process_data(train_data, test_data):
	binary_cols, all_zero_cols, all_one_cols = [],[],[]
	for col in train_data.iloc[:,10:]:
		unique_vals = train_data[col].unique()
		if np.array_equal(unique_vals, [1,0]) or np.array_equal(unique_vals, [0,1]):
			binary_cols.append(col)
		elif np.array_equal(unique_vals, [0]):
			all_zero_cols.append(col)
		elif np.array_equal(unique_vals, [1]):
			all_one_cols.append(col)
		else:
			print(unique_vals)

	# Drop columns with only zeros
	train_data = train_data.drop(all_zero_cols, axis=1)
	test_data = test_data.drop(all_zero_cols, axis=1)

	train_cat_cols = train_data.iloc[:,2:10]
	test_cat_cols = test_data.iloc[:,1:9]
	freq=[]
	col_names = []
	cat_mismatch = []

	for train_col, test_col in zip(train_cat_cols, test_cat_cols):
		col_names.append(train_col)
		train_freq = len(train_cat_cols[train_col].unique())
		test_freq = len(test_cat_cols[test_col].unique())

		if train_freq!=test_freq:
			cat_mismatch.append(train_col)

		freq.append([train_freq, test_freq])
	freq = pd.DataFrame(freq, columns=['Train_Freq', 'Test_Freq'], index=col_names)

	train_data = train_data.drop(cat_mismatch, axis=1)
	test_data = test_data.drop(cat_mismatch, axis=1)

	return train_data, test_data


def prepare_data_ml(train_data, test_data):
	X_train = pd.get_dummies(train_data)
	X_train = X_train.drop(['ID','y'], axis=1).values
	y_train = train_data.y.values

	X_test = pd.get_dummies(test_data)
	y_test_id = test_data.ID.values
	X_test = X_test.drop(['ID'], axis=1).values

	return X_train, y_train, X_test, y_test_id


def gridSearch_gbm(X_train, y_train, K=5):
	param_grid = param_grid = {'n_estimators': [100,500,1000], 'max_depth': [3,5,10],
								'min_samples_split': [2],
								'learning_rate':[10e-3, 10e-2,10e-1] ,
								'loss': ['ls','huber'],
								'subsample':[1], 'max_features':['sqrt','log2'],
								'criterion':['friedman_mse']}

	cv_kfold = KFold(n_splits=K, shuffle=True, random_state=12548)

	gbm_regressor = ensemble.GradientBoostingRegressor(random_state=122)

	gs_gbm = GridSearchCV(estimator=gbm_regressor,
	                      param_grid=param_grid,
	                      scoring='r2',
	                      cv=cv_kfold,
	                      n_jobs=-1,
	                      verbose=1)
	gs_gbm.fit(X_train, y_train)
	print(gs_gbm.best_score_)

	return gs_gbm.best_estimator_


def eval_gbm(gbm_regressor, X_train, y_train, K=5):
	cv_ss = ShuffleSplit(n_splits=K, test_size=0.3, random_state=12548)
	reg_scores = []

	for train_idx, val_idx in cv_ss.split(X_train, y_train):
		x_train_cv, y_train_cv = X_train[train_idx], y_train[train_idx]
		x_val_cv, y_val_cv = X_train[val_idx], y_train[val_idx]

		reg_model = gbm_regressor
		reg_model.fit(x_train_cv, y_train_cv)
		y_pred_cv = reg_model.predict(x_val_cv)

		r_2 = mt.r2_score(y_val_cv, y_pred_cv) # Coefficient of determination
		mse = mt.mean_squared_error(y_val_cv, y_pred_cv) # Mean squared error
		explained_var = mt.explained_variance_score(y_val_cv, y_pred_cv) # Explained variance

		reg_scores.append([r_2, mse, explained_var])

	reg_scores = pd.DataFrame(reg_scores, columns=['R^2','MSE','Explained_Variance'])
	return reg_scores


def make_submission(reg_estimator, X_test, ID, fname='FinalSubmission'):
    y_pred = reg_estimator.predict(X_test)
    final_submission = pd.DataFrame(np.hstack([ID[:,np.newaxis], y_pred[:,np.newaxis]]), columns=['ID','y'])
    final_submission.ID = final_submission.ID.astype(int)
    final_submission.to_csv('./results/'+fname, index=False)
    return final_submission


def run_gridSearch_gbm():
	print("Read and preprocess data")

	train_data, test_data = read_data()
	train_data, test_data = process_data(train_data, test_data)

	X_train, y_train, X_test, y_test_id = prepare_data_ml(train_data, test_data)

	print("Training Samples: ", X_train.shape)
	print("Test Sample: ", X_test.shape)

	print("============ Grid Search GBM ================")

	gbm_best_estimator = gridSearch_gbm(X_train, y_train, K=3)
	print(gbm_best_estimator)


if __name__ == '__main__':
	start_time = time.time()
	run_gridSearch_gbm()
	print("----%s seconds----"%(time.time()-start_time))
