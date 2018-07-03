import pandas as pd

# read the raw data file from the directory
raw_data = pd.read_csv("Building_Permits.csv", low_memory=False)

# drop the columns not necessary for this assignment
raw_data = raw_data.drop(['MOST_RECENT_INSPECTION', 'NUMBER_OF_STORIES', 'SQFT10',
                            'SQFT25', 'SQFM30', 'SQFO30', 'INTERNAL_AREA',
                            'EXIST_RES_UNITS', 'NEW_RES_UNITS', 'INTERNAL_AREA',
                            'TOTAL_SQ_FOOTAGE', 'ELECTORAL_DISTRICT', 'CIVIC_ID',
                            'CIVIC_NUMBER', 'STREET_NAME', 'STREET_TYPE',
                            'SHORT_PROJECT_DESCRIPTION', 'LONG_PROJECT_DESCRIPTION'], axis=1)

# drop the rows that have NA values in them
cleaned_data = raw_data.dropna(subset=['DATE_OF_APPLICATION'], how='any')

""" for NAN's in ALTERNATE_BUILDING_TYPE column we will be removing when
we will be working with visualizations in Tableau software. There are so many
NAN values in this column that if we remove rows corresponding to them, our
dataset will be reduced to 1/8th of its data"""

# remove unwanted character from Date and time columns
cleaned_data["DATE_OF_APPLICATION"] = cleaned_data["DATE_OF_APPLICATION"].astype(str).apply(lambda x: x.replace("T", " "))
cleaned_data["DATE_OF_PERMIT_ISSUANCE"] = cleaned_data["DATE_OF_PERMIT_ISSUANCE"].astype(str).apply(
    lambda x: x.replace("T", " "))

# conversion to datetime datatype
cleaned_data["DATE_OF_APPLICATION"] = pd.to_datetime(raw_data["DATE_OF_APPLICATION"], format="%Y-%m-%d %H:%M:%S")
cleaned_data["DATE_OF_PERMIT_ISSUANCE"] = pd.to_datetime(raw_data["DATE_OF_PERMIT_ISSUANCE"], format="%Y-%m-%d %H:%M:%S")

# some necessary datatype conversions
cleaned_data["PERMIT_NUMBER"] = cleaned_data["PERMIT_NUMBER"].astype(int)
cleaned_data['ESTIMATED_VALUE_OF_PROJECT'] = cleaned_data['ESTIMATED_VALUE_OF_PROJECT'].replace(0, cleaned_data[
    'ESTIMATED_VALUE_OF_PROJECT'].mean())

# store the cleaned dataset in a CSV file
cleaned_data.to_csv('clean_data.csv', header=True, index=False, sep=',')