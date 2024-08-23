from itertools import islice
import re
import os

from gluonts.dataset.pandas import PandasDataset

import pandas as pd
import numpy as np

from scripts.utils import convertcolumns_tofloat, convert_to_date


def check_and_modify_2020_year(df_jobs):

    columns = df_jobs.columns
    
    for category in df_jobs['North American Industry Classification System (NAICS)'].unique():
        category_df = df_jobs[df_jobs['North American Industry Classification System (NAICS)'] == category]
        
        if 2020 in category_df.index.year:
            year_2020_df = category_df[category_df.index.year == 2020]
            
            if len(year_2020_df) < 12:
                all_months = pd.date_range(start='2020-01-01', end='2020-12-01', freq='MS')
                missing_months = set(all_months) - set(year_2020_df.index)
                
                for month in missing_months:
                    # Create a new row with the same columns as the original DataFrame
                    new_row = pd.DataFrame(index=[month], columns=columns)
                    new_row['North American Industry Classification System (NAICS)'] = category
                    new_row['VALUE'] = None
                    # Ensure the new row has the same dtypes as the original DataFrame
                    new_row = new_row.astype(df_jobs.dtypes)
                    
                    df_jobs = pd.concat([df_jobs, new_row])
    
    df_jobs = df_jobs.sort_index()
    return df_jobs


def No_external_load_dataset(path_to_csv):
    df_earnings_ = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)
    df_earnings_ = df_earnings_[df_earnings_['GEO']=='Canada']
    
    if 'Statistics' in df_earnings_.keys():
        print('Job Vacancies Time')
        df_earnings_ = df_earnings_[df_earnings_['Statistics'] == 'Job vacancies']
        df_earnings_ = df_earnings_[['North American Industry Classification System (NAICS)', 'VALUE']]
        # df_earnings_ = df_earnings_[df_earnings_['North American Industry Classification System (NAICS)'] == 'Total, all industries']
        df_earnings_ = check_and_modify_2020_year(df_earnings_)

    # df_earnings_ = df_earnings_[df_earnings_['North American Industry Classification System (NAICS)'] == 'Total, all industries']
    df_earnings_ = df_earnings_[['North American Industry Classification System (NAICS)', 'VALUE']]

    convertcolumns_tofloat(df_earnings_)
    df_earnings_
    # # # Split the DataFrame into train and test sets
    earnings_train_df = df_earnings_[df_earnings_.index <= '2022-06']
    earnings_val_df = df_earnings_[df_earnings_.index <= '2023-6']
    earnings_test_df = df_earnings_
    # # # Create the Pandas
    train_earnings_dataset = PandasDataset.from_long_dataframe(earnings_train_df, target="VALUE", item_id="North American Industry Classification System (NAICS)")
    val_earnings_dataset = PandasDataset.from_long_dataframe(earnings_val_df, target="VALUE", item_id="North American Industry Classification System (NAICS)")
    test_earnings_dataset = PandasDataset.from_long_dataframe(earnings_test_df, target="VALUE", item_id="North American Industry Classification System (NAICS)")

    return train_earnings_dataset, val_earnings_dataset, test_earnings_dataset

def with_external_load_dataset(path_to_csv, selected_features=None):
    df_earnings_ = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)
    df_earnings_['feature_value'] = df_earnings_['feature_value'].replace(0, np.nan)
    columns_to_fill = df_earnings_.columns.difference(['feature_value'])
    df_earnings_[columns_to_fill] = df_earnings_[columns_to_fill].fillna(0)
    convertcolumns_tofloat(df_earnings_)


    # Split the DataFrame into train and test sets
    earnings_train_df = df_earnings_[df_earnings_. index <= '2023-08'] # original '2022-06'
    earnings_val_df = df_earnings_[df_earnings_.index <= '2023-11'] # '2022-12'
    earnings_test_df = df_earnings_[df_earnings_.index < '2024-12'] # '2023-12' 
    
    # Create the Pandas
    
        # Get feature columns other than "feature_value" and 'feature_name'
    if selected_features is not None:
        feat_columns = selected_features
    else:
        feat_columns = [col for col in df_earnings_.columns if col not in ["feature_value", "feature_name"]]
    
    
    train_earnings_dataset = PandasDataset.from_long_dataframe(earnings_train_df, target="feature_value", item_id = 'feature_name', feat_dynamic_real = feat_columns)
    val_earnings_dataset = PandasDataset.from_long_dataframe(earnings_val_df, target="feature_value", item_id = 'feature_name', feat_dynamic_real = feat_columns)
    test_earnings_dataset = PandasDataset.from_long_dataframe(earnings_test_df, target="feature_value", item_id = 'feature_name', feat_dynamic_real = feat_columns)

    return train_earnings_dataset,val_earnings_dataset, test_earnings_dataset


#any additional functions that can be used to help with input files in the time_series_conversion function.




#The dataset must be correctly formatted/aligned to the monthly format in the base files, and has identify date, category, and value variables for the pivot
# file path: self explantory
# times steps: 20
# var_name: dataset name appended to each variable
# date_variable: the date variable name
# category_variable: categorization of value variables
# value_variables: self explantory
def time_series_conversion(file_path,var_name,date_variable,category_variables,values_variables,time_steps=20):

    print(file_path)

    #file input 
    if file_path == 'Business_Outlook_Survey-ed-2023-10-01.csv':
        df = pd.read_csv(file_path,skiprows=57)
    else:
        df = pd.read_csv(file_path)

    #specific instructions for certain files
    if file_path == 'Employment_ByIndustry.csv':
        df = df[df['GEO']=='Canada']
    elif file_path == 'Business_Outlook_Survey-ed-2023-10-01.csv':
        df[date_variable[0]] = df[date_variable[0]].apply(convert_to_date)
        df = df[df[date_variable[0]]>= '2001-01-01']
        df[date_variable[0]] = pd.to_datetime(df[date_variable[0]])
        df = df.set_index(date_variable[0])
        df = df.shift(1, freq='D').resample('MS').bfill()
        df.reset_index(inplace=True)
    elif file_path == '36100434.csv':
        df[date_variable[0]] = pd.to_datetime(df[date_variable[0]], format='%Y-%m').dt.strftime('%Y-%m-%d')
        df = df[(df['North American Industry Classification System (NAICS)'] == 'All industries [T001]') & 
                (df['Seasonal adjustment'] == 'Seasonally adjusted at annual rates') &
                (df['Prices'] == '2017 constant prices') &
                (df[date_variable[0]]>= '2001-01-01')]
    elif file_path == '18100004.csv':
        df[date_variable[0]] = pd.to_datetime(df[date_variable[0]], format='%Y-%m').dt.strftime('%Y-%m-%d')
        df = df[(df['Products and product groups'] == 'All-items') & 
                (df['GEO'] == 'Canada') &
                (df[date_variable[0]]>= '2001-01-01')]
    elif file_path == '10100139.csv':
        df[date_variable[0]] = pd.to_datetime(df[date_variable[0]])
        df = df[df[date_variable[0]].dt.day == 1]
        df = df[((df['Financial market statistics'] == 'Bank rate' ) | 
                (df['Financial market statistics'] == 'Target rate' )) &
                (df[date_variable[0]]>= '2001-01-01')]
        df[date_variable[0]] = df[date_variable[0]].dt.strftime('%Y-%m-%d')
    elif file_path == 'covid19-download.csv':
        df[date_variable[0]] = pd.to_datetime(df[date_variable[0]])
        df = df[df['prname'] == "Canada"]
        df = df.groupby(df[date_variable[0]].dt.to_period('M')).agg({
            date_variable[0]: 'first',
            'prname': 'first', 
            'numtotal_last7': 'sum'})
        df.reset_index(drop=True, inplace=True)
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df[date_variable[0]] = df[date_variable[0]].dt.strftime('%Y-%m-%d')
    elif file_path == 'Hiring_Lab_full_country_data_to_2024-03-08.csv':
        df = df[(df['jobcountry']=='Canada') &
                (df['Postings Type']=='Overall Postings')&
                (df['Index Type']=='Indeed Job Postings Index, Feb 01 2020 = 100')]
        df[date_variable[0]] = pd.to_datetime(df['Date'])
        df = df[df[date_variable[0]].dt.is_month_start]
        df[date_variable[0]] = df[date_variable[0]].dt.strftime('%Y-%m-%d')
    elif file_path == '17100009.csv':
        df[date_variable[0]] = pd.to_datetime(df[date_variable[0]], format='%Y-%m').dt.strftime('%Y-%m-%d')
        df = df[(df[date_variable[0]]>= '2001-01-01')&(df['GEO'] == 'Canada')]
        df[date_variable[0]] = pd.to_datetime(df[date_variable[0]])
        df = df.set_index(date_variable[0])
        df = df.shift(1, freq='D').resample('MS').bfill()
        df.reset_index(inplace=True)
    
    #filter on pivot variables and then pivot dataframe accordingly
    df = df[date_variable + category_variables + values_variables]
    if len(category_variables)>0:
        df = df.pivot(index=date_variable,columns = category_variables,values=values_variables).reset_index()
        #... also fix multi index issues
        for c in category_variables:
            df = df.reset_index(drop=True)
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
        df.rename(columns= {df.columns[0]:'REF_DATE'},inplace=True)
    df[df.columns[0]]=pd.to_datetime(df[df.columns[0]]).dt.strftime('%Y-%m-%d')
    df.set_index(df.columns[0], inplace=True)

    #fix add timestep columns
    for col in df:
        for lag in range(1, time_steps + 1):
            df[f'{col}_t-{lag}'] = df[col].shift(lag)

    #add id to columns so that they don't overlap with other files
    df.columns = var_name + df.columns

    return df

# Define a function to extract timesteps and handle columns without timesteps
def extract_timestep(column_name):

    timestep_pattern = re.compile(r"_t-(\d+)$")

    match = timestep_pattern.search(column_name)
    if match:
        # Return the timestep as an integer to sort numerically
        return int(match.group(1))
    else:
        # Assign a default value that sorts before the timesteps (e.g., -1)
        return -1

def data_creation(time_steps):

    #base data
    df_job = time_series_conversion('JobVacancies_ByIndustry.csv','job_',['REF_DATE'],['North American Industry Classification System (NAICS)','Statistics'],['VALUE'],time_steps)
    df_ear = time_series_conversion('Earnings_ByIndustry.csv','ear_',['REF_DATE'],['North American Industry Classification System (NAICS)'],['VALUE'],time_steps)
    df_emp = time_series_conversion('Employment_ByIndustry.csv','emp_',['REF_DATE'],['North American Industry Classification System (NAICS)'],['VALUE'],time_steps)
    df_hou = time_series_conversion('Hours_ByIndustry.csv','hou_',['REF_DATE'],['North American Industry Classification System (NAICS)'],['VALUE',],time_steps)
    df_bus = time_series_conversion('Business_Outlook_Survey-ed-2023-10-01.csv','bus_',['date'],[],['EMPLOY','COSTS','OUTPUTS'],time_steps)
    df_gdp = time_series_conversion('36100434.csv','gdp_',['REF_DATE'],[],['VALUE'],time_steps)
    df_cpi = time_series_conversion('18100004.csv','cpi_',['REF_DATE'],[],['VALUE'],time_steps)
    df_inf = time_series_conversion('10100139.csv','inf_',['REF_DATE'],['Financial market statistics'],['VALUE'],time_steps)
    df_pop = time_series_conversion('17100009.csv','pop_',['REF_DATE'],['GEO'],['VALUE'],time_steps)
    df_covid = time_series_conversion('covid19-download.csv','covid_',['date'],['prname'],['numtotal_last7'],time_steps)
    df_indeed = time_series_conversion('Hiring_Lab_full_country_data_to_2024-03-08.csv','indeed_',['Date'],['Sector'],['Non-seasonally adjusted percentage'],time_steps)
    output_var_size = sum([len(dfs.columns) for dfs in [df_job, df_ear, df_emp, df_hou]])/(time_steps+1)
    input_var_size = sum([len(dfs.columns) for dfs in [df_pop,df_indeed,df_covid,df_inf,df_cpi,df_gdp,df_bus,df_job, df_ear, df_emp, df_hou]])/(time_steps+1)
    df = df_pop
    for df1 in [df_covid,df_indeed,df_inf,df_cpi,df_gdp,df_bus,df_job,df_ear,df_emp,df_hou]:
        df = pd.merge(df,df1,how = 'outer',left_index=True,right_index=True)

    # Sort the DataFrame's columns by the extracted timestep
    sorted_columns = sorted(df.columns, key=extract_timestep,reverse=True)
    df = df[sorted_columns]
    df = df[(df.index >= '2000-02-01') & (df.index <= '2023-12-01')]
    df.fillna(0,inplace=True)
    
    return df_job,df_ear,df_emp,df_hou,df,input_var_size,output_var_size

def convert_to_csv(df,input_var_size):
    labour_ind = ['job_','ear_','emp_','hou_']
    for ind in labour_ind:
        if ind == 'job_':
            df_col = [col for col in df.columns if col.startswith(ind) and col.endswith('cies')]
            other_col = [col for col in df.columns if not(col.startswith(ind)) or (col.startswith(ind) and not(col.endswith('cies')))]
        else:
            df_col = [col for col in df.columns if col.startswith(ind)]
            other_col = [col for col in df.columns if not(col.startswith(ind))]
        df[other_col] = df[other_col].shift(1)
        df.fillna(0,inplace=True)
        melt_df = df.iloc[:,-int(input_var_size):][df_col]
        melt_df = melt_df.reset_index()
        melt_df = pd.melt(melt_df, id_vars=['index'], var_name='feature_name', value_name='feature_value')
        melt_df = melt_df.set_index('index')
        melt_df.head()
        df_gluton = pd.merge(melt_df,df,how = 'left',left_index=True,right_index=True)
        df_gluton.to_csv(f'{ind}melt_complete_data.csv')