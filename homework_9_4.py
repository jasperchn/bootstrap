
import time
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 20:44:01 2019

@author: Sheryl Zhang
"""
import datetime
# import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.optimize import minimize
import json
# import matplotlib.pyplot as plt

random_list = np.random.uniform(0,1,100)


class Fast_cdf_norm(object):
	"""docstring for Fast_cdf_norm"""
	def __init__(self):
		self.base_X=np.linspace(-5,5,10000)

		# print("self.base_X",self.base_X)
		self.base_cdf_result=np.array([norm.cdf(x) for x in self.base_X])
		self.base_ppf_result=np.array([norm.cdf(x) for x in self.base_X])
		# plt.plot(self.base_X,self.base_cdf_result)
		# plt.show()
		# print(self.base_X)
	def fast_cdf(self,x):
		location=((x-(-5))/10)*(len(self.base_X)-1)
		location_int=max(min(int(location),len(self.base_X)-1),0)
		location_float=location-location_int
		# print("location_int,location_float",location_int,location_float)
		if location_int==0:
			return self.base_cdf_result[0]
		elif location_int==len(self.base_X)-1:
			return self.base_cdf_result[-1]
		else:

			

			# result=self.base_cdf_result[location_int-1]+location_float*(self.base_cdf_result[location_int]-self.base_cdf_result[location_int-1])
			result=self.base_cdf_result[location_int]*(1-location_float) +location_float*self.base_cdf_result[location_int+1]
			# print("x,location_int,ocation_float",x,location_int,location_float)
			# result=self.base_cdf_result[location_int]
		return result

	def fast_ppf(self,x):
		location=((x-(-5))/10)*(len(self.base_X)-1)
		location_int=max(min(int(location),len(self.base_X)-1),0)
		location_float=location-location_int
		# print("location_int,location_float",location_int,location_float)
		if location_int==0:
			return self.base_ppf_result[0]
		elif location_int==len(self.base_X)-1:
			return self.base_ppf_result[-1]
		else:

			result=self.base_ppf_result[location_int]*(1-location_float) +location_float*self.base_ppf_result[location_int+1]
		return result


My_fast_cdf_norm=Fast_cdf_norm()



def data_restructure():
	# in this function, I will restructure the dataframe, I will add 
	# two indicator to indicate whether it is a product 1 or product 2
	# xls = pd.ExcelFile('purchase.xlsx')
	# purchase_data = pd.read_excel(xls, 'Purchase_data')
	purchase_data = pd.read_csv("purchase.csv")
	purchase_data = purchase_data.rename(columns={"Purchase Product 1":"Purchase1","log(price of product 1)":"price1",\
		"Purchase Product 2":"Purchase2","log(price of product 2)":"price2"})	
	purchase_data["purchase_both"] = purchase_data["Purchase1"]*purchase_data["Purchase2"]
	purchase_data["purchase_1_only"] = purchase_data["Purchase1"]*(1-purchase_data["Purchase2"])
	purchase_data["purchase_2_only"] = (1-purchase_data["Purchase1"])*purchase_data["Purchase2"]
	purchase_data["no_purchase"] = (1-purchase_data["Purchase1"])*(1-purchase_data["Purchase2"])
	purchase_data["purchase_identify"]= purchase_data.apply(lambda x: json.dumps([x["purchase_both"],x["purchase_1_only"],x["purchase_2_only"],x["no_purchase"]]),axis=1)
	purchase_data["purchase_identify"] = purchase_data["purchase_identify"].apply(lambda x: json.loads(x))

	return purchase_data	



def prob_calculate(price1,price2,purchase_identify,alpha_1,alpha_2,lamb,gamma21,num_iter):
	start=time.clock()
	# random_list = np.random.uniform(0,1,num_iter) 
	# print("t11",time.clock()-start)
	# eta_1 = [norm.ppf(x+(1-x)*My_fast_cdf_norm.fast_cdf(-alpha_1-lamb*price1)) for x in random_list]
	eta_1 = [My_fast_cdf_norm.fast_ppf(x+(1-x)*My_fast_cdf_norm.fast_cdf(-alpha_1-lamb*price1)) for x in random_list]
	# eta_1 = [My_fast_cdf_norm.fast_ppf(x*0.5+0.5) for x in random_list]

	# print("t12",time.clock()-start)
	# eta_1 = truncated_norm_simulate(price1,price2,alpha_1,alpha_2,lamb,gamma21,num_iter)
	# print("11",time.clock()-start)
	phi_1 = [(1-My_fast_cdf_norm.fast_cdf(-alpha_1-lamb*price1)) for i in range(num_iter)]
	# print("t13",time.clock()-start)
	phi_2 = [(1 - My_fast_cdf_norm.fast_cdf(-alpha_2-lamb*price2-gamma21*x)) for x in eta_1]
	# print("t14",time.clock()-start)
	# phi_1 = [(1-My_fast_cdf_norm.fast_cdf(-alpha_1-lamb*price1)) for i in range(num_iter)]
	# phi_2 = [(1 - My_fast_cdf_norm.fast_cdf(-alpha_2-lamb*price2-gamma21*x)) for x in eta_1]
	# print("12",time.clock()-start)
	prob_both_list = [phi_1[i]*phi_2[i] for i in range(len(phi_1))]
	prob_1_only_list = [phi_1[i]*(1-phi_2[i]) for i in range(len(phi_1))]
	prob_2_only_list = [(1-phi_1[i])*phi_2[i] for i in range(len(phi_1))]
	prob_no_list = [(1-phi_1[i])*(1-phi_2[i]) for i in range(len(phi_1))]

	prob_list = [sum(np.array(purchase_identify)*np.array([prob_both_list[i],prob_1_only_list[i],prob_2_only_list[i],prob_no_list[i]])) for i in range(len(prob_both_list))]
	# print("13",time.clock()-start)
	prob = (1/num_iter)*sum(prob_list)
	# print("t15",time.clock()-start)
	return prob

def Likelihood_func(params):
	global global_count
	global global_start
	global_count+=1
	print(global_count,time.clock()-global_start,params)
	start=datetime.datetime.now()

	alpha_1,alpha_2,lamb,gamma21=params[0],params[1],params[2],params[3]
	df = data_restructure()
	# print("t1",datetime.datetime.now()-start)
	df["prob"] = df.apply(lambda x: prob_calculate(x["price1"],x["price2"],x["purchase_identify"],alpha_1,alpha_2,lamb,gamma21,num_iter=100),axis=1)
	# print("t2",datetime.datetime.now()-start)
	# print(df["prob"])
	df["log_prob"] = np.log(df["prob"])
	neg_ll = -np.sum(df["log_prob"])
	# print("t3",datetime.datetime.now()-start)
	print("neg_ll",neg_ll)
	return neg_ll

def params_estimate():
	global global_start
	global global_count
	global_count=0
	global_start=time.process_time()
	guess = np.array([ 1, 0.5, -0.3,  0.5])
	# print("Likelihood_func(guess)",Likelihood_func(guess))
	# quit()
	results = minimize(Likelihood_func,guess,method = 'Nelder-Mead',tol = 0.01, options={'maxiter':100, 'disp':True})
	
	estimated_parameters = results.x
	print("results",results)

	return results


if __name__ == '__main__':
	# start = time.clock()
	# data_restructure()
	# prob_calculate(price1=1,price2=1,alpha_1=1,alpha_2=1,lamb=1,gamma21=1,num_iter=50)	
	params_estimate()
	# Likelihood_func([1,1,0,1])
	# end = time.clock()
	# print(end-start)