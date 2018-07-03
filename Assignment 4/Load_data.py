import pandas as pd

raw_data = pd.read_csv("Building_Permits.csv", low_memory=False)

raw_data[pd.to_numeric(raw_data.PERMIT_NUMBER, errors='coerce').notnull()]

raw_data = raw_data.drop(['MOST_RECENT_INSPECTION', 'NUMBER_OF_STORIES'], axis=1)

raw_data = raw_data.dropna(subset=['DATE_OF_APPLICATION', 'LONG_PROJECT_DESCRIPTION',
                                   'SHORT_PROJECT_DESCRIPTION'], how='any')

final_data = raw_data.drop(['SQFT10', 'SQFT25', 'SQFM30', 'SQFO30', 'INTERNAL_AREA',
                            'EXIST_RES_UNITS', 'NEW_RES_UNITS', 'INTERNAL_AREA',
                            'TOTAL_SQ_FOOTAGE', 'ELECTORAL_DISTRICT', 'CIVIC_ID',
                            'CIVIC_NUMBER', 'STREET_NAME', 'STREET_TYPE',
                            'SHORT_PROJECT_DESCRIPTION', 'LONG_PROJECT_DESCRIPTION'], axis=1)

final_data["DATE_OF_APPLICATION"] = final_data["DATE_OF_APPLICATION"].astype(str).apply(lambda x: x.replace("T", " "))
final_data["DATE_OF_PERMIT_ISSUANCE"] = final_data["DATE_OF_PERMIT_ISSUANCE"].astype(str).apply(
    lambda x: x.replace("T", " "))

final_data["DATE_OF_APPLICATION"] = pd.to_datetime(raw_data["DATE_OF_APPLICATION"], format="%Y-%m-%d %H:%M:%S")
final_data["DATE_OF_PERMIT_ISSUANCE"] = pd.to_datetime(raw_data["DATE_OF_PERMIT_ISSUANCE"], format="%Y-%m-%d %H:%M:%S")

final_data["PERMIT_NUMBER"] = final_data["PERMIT_NUMBER"].astype(int)

final_data['ESTIMATED_VALUE_OF_PROJECT'] = final_data['ESTIMATED_VALUE_OF_PROJECT'].replace(0, final_data[
    'ESTIMATED_VALUE_OF_PROJECT'].mean())

final_data.to_csv('clean_data.csv', header=True, index=False, sep=',')