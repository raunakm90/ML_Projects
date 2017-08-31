# Analyze Charlotte food/restuarant/nightlife business

# Import required modules
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'D:\\Yelp\\Data\\')

# Analyze Charlotte food/restuarant/nightlife business

class business():
    def __init__(self):
            self.raw_data = pd.read_csv(".\Data\yelp_academic_dataset_business.csv")
            # Convert selected columns to boolean type
            self.col_dict_bool = {'attributes.Accepts Credit Cards':np.bool,'attributes.Accepts Insurance':np.bool,
        'attributes.Ambience.casual':np.bool,'attributes.Ambience.classy':np.bool,'attributes.Ambience.divey':np.bool,
        'attributes.Ambience.hipster':np.bool,'attributes.Ambience.intimate':np.bool,
        'attributes.Ambience.romantic':np.bool,'attributes.Ambience.touristy':np.bool,
        'attributes.Ambience.trendy':np.bool, 'attributes.Ambience.upscale':np.bool,'attributes.BYOB':np.bool,
        'attributes.By Appointment Only':np.bool,
        'attributes.Caters':np.bool,'attributes.Coat Check':np.bool,'attributes.Corkage':np.bool,
        'attributes.Delivery':np.bool,'attributes.Dietary Restrictions.dairy-free':np.bool,
        'attributes.Dietary Restrictions.gluten-free':np.bool,
        'attributes.Dietary Restrictions.halal':np.bool,'attributes.Dietary Restrictions.kosher':np.bool,
        'attributes.Dietary Restrictions.soy-free':np.bool,'attributes.Dietary Restrictions.vegan':np.bool,
        'attributes.Dietary Restrictions.vegetarian':np.bool,'attributes.Dogs Allowed':np.bool,
        'attributes.Drive-Thru':np.bool,'attributes.Good For Dancing':np.bool,'attributes.Good For Groups':np.bool,
        'attributes.Good For.breakfast':np.bool,'attributes.Good For.brunch':np.bool,
        'attributes.Good For.dessert':np.bool,'attributes.Good For.dinner':np.bool,
        'attributes.Good For.latenight':np.bool,'attributes.Good For.lunch':np.bool,'attributes.Good for Kids':np.bool,
        'attributes.Happy Hour':np.bool,'attributes.Has TV':np.bool,'attributes.Music.background_music':np.bool,
        'attributes.Music.dj':np.bool,'attributes.Music.jukebox':np.bool,
        'attributes.Music.karaoke':np.bool,'attributes.Music.live':np.bool,
        'attributes.Music.video':np.bool,'attributes.Open 24 Hours':np.bool,'attributes.Order at Counter':np.bool,
        'attributes.Outdoor Seating':np.bool,'attributes.Parking.garage':np.bool,'attributes.Parking.lot':np.bool,
        'attributes.Parking.street':np.bool,'attributes.Parking.valet':np.bool,'attributes.Parking.validated':np.bool,
        'attributes.Take-out':np.bool,'attributes.Takes Reservations':np.bool,'attributes.Waiter Service':np.bool,
        'attributes.Wheelchair Accessible' :np.bool, 'open':np.bool}
            map_dict = {0:False, 1:True}
            for col in self.raw_data.columns:
                    if col in self.col_dict_bool.keys():
                            self.raw_data[col] = self.raw_data[col].map(map_dict)
    def city_subset(self,city = 'Charlotte',categories = ['Food','Restaurants','Nightlife']):
        subset = self.raw_data.loc[self.raw_data['city'] == city]
        self.city_data = pd.DataFrame()
        for index,row in subset.iterrows():
            if any(category in row.categories for category in categories):
                self.city_data = self.city_data.append(row)
        return self

    def remove_na(self,axis = 1):
        self.city_data = self.city_data.dropna(axis = axis, how = 'all')
        return self

    def get_neighborhoods(self):
        neighborhoods_1 = list()
        neighborhoods_2 = list()
        for item in food_charlotte_business.neighborhoods:
            ngh = item.strip('[]')
            ngh = ngh.strip().split(',')
        if (len(ngh) == 1) and ngh[0] != "":
            neighborhoods_1.append(ngh[0].strip("'"))
            neighborhoods_2.append(np.nan)
        elif (len(ngh) == 2):
            neighborhoods_1.append(ngh[0].strip("'"))
            neighborhoods_2.append(ngh[1].strip("'"))
        else:
            neighborhoods_1.append(np.nan)
            neighborhoods_2.append(np.nan)
        self.city_data['neighborhoods_1'] = pd.Series(neighborhoods_1,
                                                      index = self.city_data.index)
        self.city_data['neighborhoods_2'] = pd.Series(neighborhoods_2,
                                                      index = self.city_data.index)
        return self

    def clean_data(self,na_val = 70,drop_columns = ['neighborhoods','city','type']):
        # Remove duplicate ID's
        self.city_data.drop_duplicates(subset = ['business_id'],keep = False,inplace = True)
        na_perc = {}
        for col in self.city_data.columns:
            na_perc[col] = self.city_data[col].isnull().sum()*100/len(self.city_data)
        for key,value in na_perc.items():
            if value > na_va:
                self.city_data.drop(key,axis = 1,inplace=True)

        # Remove unwanted columns
        self.city_data.drop(drop_columns,axis = 1,inplace = True)
        return self

    def fillna_attr(self):
        col_list = []
        for name in self.city_data.columns:
            if 'attributes' in name and name in self.col_dict_bool.keys():
                col_list.append(name)
        self.city_data[col_list].fillna(0,inplace=True)
        return self

    def get_working_min(self):
        weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        discard_columns = []
        for day in weekday:
            temp = [col for col in self.city_data.columns if day in col]
            discard_columns.append(temp[0])
            discard_columns.append(temp[1])
            self[day] = abs(pd.to_datetime(self[temp[1]])-
                            pd.to_datetime(self[temp[0]])).astype('timedelta64[m]')
        discard_columns.append('city')
        discard_columns.append('type')
        discard_columns.append('full_address')
        discard_columns.append('neighborhoods')
        discard_columns.append('neighborhoods_2')
        discard_columns.append('state')

        return (self.city_data.drop(discard_columns,axis = 1, inplace = True))

class reviews():
	def __init__(self):
		self.raw_data = pd.read_csv('./Data/yelp_academic_dataset_review.csv')
		self.raw_data.date = pd.to_datetime(self.raw_data['date'])
		self.raw_data.drop('type',axis = 1, inplace = True)

class users():
	def __init__(self):
		self.raw_data = pd.read_csv('./Data/yelp_academic_dataset_user.csv')

class checkin():
	def __init__(self):
		self.raw_data = pd.read_csv('./Data/yelp_academic_dataset_checkin.csv')







