'@Author - Raunak Mundada'
'Date - 9/5/2016'
###############################################################
# Set up MonogDB database for Yelp
# Dump all json files into MongoDB
# Run MongoDB instance before running this program
# mongod --directoryperdb --dbpath D:/Yelp/Data/db
# --logpath D:/Yelp/Data/log/mongodb.log --logappend
################################################################


import json
import glob
from pymongo import MongoClient

files_path = 'D:/Yelp/Data/*.json'

#source: http://stackoverflow.com/questions/30088006/cant-figure-out-how-to-fix-the-error-in-the-following-code
def read_json_ed(path):
    with open(path, 'rb') as f:
        data = f.readlines()
    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)
    # each element of 'data' is an individual JSON object
    # i want to convert it into an *array* of JSON objects
    # which, in and of itself, is one large JSON object
    # basically... add square brackets to the beginning
    # and end, and have all the individual business JSON objects
    # separated by a comma
    data_str = "[" + ','.join(data) + "]"
    data_json = json.loads(data_str)
    # now, load it into pandas
    #data_df = pd.read_json(data_json_str)
    return data_json

def create_db(data, dataset_type):

	#Create connection to mongo db client
	try:
		client = MongoClient()
		print ("Connected: Dumping JSON data files into MongoDB")
	except pymongo.errors.ConnectionFailure,e:
		print ("Could not connect to MongoDB: %s", e)

	# Accessing a database. Creates 'test_db' database automatically if it does not exist
	db = client.yelp_db #client['test_db']

	# Accessing a collection (referred to as table in relational db)
	# Documents refers to records (docs are stored as json files)
	collection = db.business #mydb['test_db']
	collection = db[dataset_type]
	#file = read_json_ed("D:/Yelp/Data/yelp_academic_dataset_business.json")
	#collection.insert(file)
	for item in data:
		collection.insert(item)

def create_db_helper():
	#print (db.business.find_one())
	files = glob.glob(files_path)
	for f in files:
		data_json = read_json_ed(f)
		create_db(data = data_json,dataset_type = data_json[0]['type'])


if __name__ == '__main__':
	try:
		client = MongoClient()
		print ("Connected Successfully!")
	except:
		print ("Connection Failed")

	if 'yelp_db' not in client.database_names():
		print ("Updating Yelp database")
		create_db_helper()
	else:
		print ("Yelp database is available")
