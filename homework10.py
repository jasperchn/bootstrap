# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 20:44:01 2019

@author: Sheryl Zhang
"""

import time
from homework_9_4 import *
import numpy
from pandas import read_csv
from sklearn.utils import resample
import scipy
from scipy import stats
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score



def resample_data(df=data_restructure()):
	df_truncated = df[["Purchase1","price1","Purchase2","price2","purchase_both","purchase_1_only","purchase_2_only","no_purchase"]]
	values = df_truncated.values
	n_size = int(len(df_truncated) * 0.50)
	sample = resample(values, n_samples=2)
	# print(sample)
	df_new = pd.DataFrame({"Purchase1":sample[:,0],"price1":sample[:,1],"Purchase2":sample[:,2],"price2":sample[:,3],"purchase_both":sample[:,4],"purchase_1_only":sample[:,5],"purchase_2_only":sample[:,6],"no_purchase":sample[:,7]})
	df_new["purchase_identify"]= df_new.apply(lambda x: json.dumps([x["purchase_both"],x["purchase_1_only"],x["purchase_2_only"],x["no_purchase"]]),axis=1)
	df_new["purchase_identify"] = df_new["purchase_identify"].apply(lambda x: json.loads(x))
	# print(len(df_new))
	return df_new

def Likelihood_func_new(params):

	alpha_1,alpha_2,lamb,gamma21=params[0],params[1],params[2],params[3]
	df = resample_data(df=data_restructure())
	# print("t1",datetime.datetime.now()-start)
	df["prob"] = df.apply(lambda x: prob_calculate(x["price1"],x["price2"],x["purchase_identify"],alpha_1,alpha_2,lamb,gamma21,num_iter=100),axis=1)
	# print("t2",datetime.datetime.now()-start)
	# print(df["prob"])
	df["log_prob"] = np.log(df["prob"])
	neg_ll = -np.sum(df["log_prob"])
	# print("t3",datetime.datetime.now()-start)
	print("neg_ll",neg_ll)
	return neg_ll

def params_estimate_new():

	guess = np.array([ 0.9, 0.4, -0.15,  0.32])
	# print("Likelihood_func(guess)",Likelihood_func(guess))
	# quit()
	results = minimize(Likelihood_func_new,guess,method = 'Nelder-Mead',tol = 0.01, options={'maxiter':100,'disp':True})
	
	estimated_parameters = list(results.x)
	# print("results",results)

	return estimated_parameters


def nonparams_bootstrap(n_boot_iter=99):
	estimated_parameters_list = []
	for i in range(n_boot_iter):
		estimated_parameters = params_estimate_new()
		estimated_parameters_list.append(estimated_parameters)
	params_array = np.array(estimated_parameters_list)
	# print(params_array)
	# print(params_array.shape)

	return params_array

def data_reparams():
	return

def params_bootstrap(n_boot_iter=99):
	return


def Calculate_se_CI(params_array,n_boot_iter=99,significance=0.9,boot_method = "non_parametric_bootstrap"):
	# params_df = nonparams_bootstrap(n_boot_iter)
	params_df = pd.DataFrame({"alpha_1":params_array[:,0],"alpha_2":params_array[:,1],"lambda":params_array[:,2],"gamma21":params_array[:,3]})
	params_df.to_csv(f"{boot_method}_result.csv")
	st_error = scipy.stats.sem(params_array)
	std = np.std(params_array,axis = 0)
	print("standard_error",st_error)
	params = [0.92, 0.44, -0.16, 0.31] # estimated params from homework9
	T_Zb = np.sqrt(n_boot_iter)*(params_array - params)/std
	print(T_Zb.shape)
	p_value_lower = ((1.0-significance)/2.0) * 100
	p_value_upper = (significance+((1.0-significance)/2.0)) * 100
	print("p_value_upper",p_value_upper)
	file = open(f'./{boot_method}_CI.txt','wb')

	for each_params in list(params_df.columns):
		params_stats = params_df[each_params]
		each_params_est = params[list(params_df.columns).index(each_params)]
		K_a = numpy.percentile(params_stats, p_value_upper)
		print("K_a",K_a)
		# lower = numpy.percentile(params_stats, p_value_lower)
		lower_bound = each_params_est- (1/np.sqrt(n_boot_iter)) * st_error[list(params_df.columns).index(each_params)] * K_a
		print("lower_bound",lower_bound)
		# upper = numpy.percentile(params_stats, p_value_upper)
		upper_bound = each_params_est + (1/np.sqrt(n_boot_iter)) * st_error[list(params_df.columns).index(each_params)] * K_a
		print("upper_bound",upper_bound)
		file.write(f'Under {boot_method}, confidence interval for {significance*100} is between {lower_bound} and {upper_bound}')



	file.close()






if __name__ == '__main__':
	Calculate_se_CI(params_array = nonparams_bootstrap(n_boot_iter=5),n_boot_iter=5)
	# outcome = params_estimate_new()
	# print("outcome",outcome)
	# resample_data()
	# from sklearn.utils import resample
	# data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	# # prepare bootstrap sample
	# boot = resample(data, replace=True, n_samples=4, random_state=1)
	# resample(data_restructure())
	
