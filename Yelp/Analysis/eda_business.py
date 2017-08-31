'@Author - Raunak Mundada'
'Date Created - 9/5/2016'

# Analyze the business data set

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pymongo
from pymongo import MongoClient

def getMongoDB_client():
	client = MongoClient() # Connect to MongoDB
	return (client.yelp_db)

def getMongoDB_query_business():
	df = list()
	yelp_db = getMongoDB_client()
	business = yelp_db.business
	doc_content = business.find({'city' : 'Charlotte'})
	for item in doc_content:
		df.append(item)
		#df.append([item['business_id'],item['stars'],item['review_count'],
		             #item['attributes'],item['categories'],item['open'],item['city']])

	return (pd.DataFrame(df))

def getMongoDB_query_review(id):
	df = list()
	yelp_db = getMongoDB_client()
	review = yelp_db.review
	doc_content = review.find({'business_id' : { '$in' : id}})

	for item in doc_content:
		df.append(item)
	return (pd.DataFrame(df))

def getMongoDB_query_tip(id):
	df = list()
	yelp_db = getMongoDB_client()
	tip = yelp_db.tip
	doc_content = tip.find({'business_id' : {'$in' : id}})

	for item in doc_content:
		df.append(item)
	return (pd.DataFrame(df))

def select_attribute(df,col,attr_key):
	x = list()
	for attr in df[col]:
		if attr_key in attr.keys():
			x.append(attr[attr_key])
		else:
			x.append(np.nan)
	return x

'''
if __name__ == '__main__':
	print ("Running Query")
	business_df = getMongoDB_query_business()
	#business_id = business_df.business_id.tolist()
	#review_df = getMongoDB_query_review(business_id)
	#print (review_df.head())
	print (business_df.head())
'''
