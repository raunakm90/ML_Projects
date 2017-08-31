'@Author - Raunak Mundada'
'Date - 9/5/2016'

##############################################################
# Download dataset from -
# https://www.yelp.com/dataset_challenge
# You need to fill in your details to get api key
# Save it in ./Data folder
# Once the data is extracted no need to run this script again

# Directory structure -
# Yelp
#  --Data
#        ----db (MongoDB database is stored here)
#        ----log (MongoDB log is stored here)
#  --Analysis
#       -----download_data.py
#       -----eda_business.py
#       -----MongoDB_setup.py
##############################################################

import argparse
import tarfile
from os.path import abspath, realpath, dirname, join as joinpath
from sys import stderr

import pandas as pd
import os

'''
def getTarFile():
	url = "https://yelp-dataset.s3.amazonaws.com/yelp_dataset_challenge_academic_dataset.tgz?Signature=AUoKrxEEJL87zyrByDeljWH3xEs%3D&Expires=1473174129&AWSAccessKeyId=AKIAJ3CYHOIAD6T2PGKA"
	print ("downloading the tar file")
	urllib.urlretrieve(url,filename = 'D:\Yelp\Data\yelp_dataset.tar')
'''
#par_dir = "D:\Yelp" # Change path according to your folder
resolved = lambda x: realpath(abspath(x))

def badpath(path, base):
    # joinpath will ignore base if path is absolute
    return not resolved(joinpath(base,path)).startswith(base)

def badlink(info, base):
    # Links are interpreted relative to the directory containing the link
    tip = resolved(joinpath(base, dirname(info.name)))
    return badpath(info.linkname, base=tip)

def safemembers(members):
    base = resolved(".")

    for finfo in members:
        if badpath(finfo.name, base):
            print >>stderr, finfo.name, "is blocked (illegal path)"
        elif finfo.issym() and badlink(finfo,base):
            print >>stderr, finfo.name, "is blocked: Hard link to", finfo.linkname
        elif finfo.islnk() and badlink(finfo,base):
            print >>stderr, finfo.name, "is blocked: Symlink to", finfo.linkname
        else:
            yield finfo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Unzip .tar file for Yelp Dataset',
            )

    parser.add_argument(
            'tar_file',
            type=str,
            help='The .tar file to unzip',
            )

    args = parser.parse_args()
    tar_filePath = args.tar_file
    #tar_filePath = "\Data\yelp_dataset_challenge_academic_dataset.tar"
    ar = tarfile.open(tar_filePath)
    ar.extractall(path = par_dir+"\Data", members=safemembers(ar))
    ar.close()
    print ("Yelp .tar file extracted to .\Data folder")
