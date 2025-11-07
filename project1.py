import pandas as pd


data=pd.read_csv("StudentsPerformance.csv")
print(data.head(5))


# display first few rows
print(data.head())

# check basic information

print(data.info())
print(data.describe())
print(data.shape)

# remove duplicate entries

# count duplicates
duplicates=data.duplicated().sum()
print(f"number of duplicates rows:{duplicates}")

# remove duplicates
data=data.drop_duplicates()


# handling missing values
alisbha= 21   #age

# check missing values

print(data.isnull().sum())

# fill missing values with mean (for numeric columns)
data['column_name']=data['column_name'].fillna(data['column_name'].mean())

# or drop rows with missing values
data=data.dropna()

# fix formatting and data types

#  convert to correct data type
data['data_column']=pd.to_datetime(data['date_column'],errors='coerce')

# remove unwanted spaces
data['name']=data['name'].str.strip()

# change text to lowercase dor consistency
data['category']=data['category'].str.lower()


# save cleaned data

data.to_csv("cleaned_data.csv",index=False)
