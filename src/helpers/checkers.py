import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import regex
import pycountry

social_media_patterns = [
    r"(?i).*facebook\.com$",   # Facebook
    r"(?i).*twitter\.com$",    # Twitter
    r"(?i).*instagram\.com$",  # Instagram
    ]

def calculate_company_name_fuzzy_score(row, column_x, column_y):
    company_name_x = str(row[column_x])  # Convert to string
    company_name_y = str(row[column_y])  # Convert to string
    
    # Handle NaN values
    if company_name_x == 'nan' or company_name_y == 'nan':
        return 0
    
    return fuzz.ratio(company_name_x, company_name_y)

def check_category_inclusion(row):
    result = np.nan
    category_x = str(row['category_x']) # Convert to string
    category_y = str(row['category_y']) # Convert to string
    if (category_x == 'nan' and category_y == 'nan') or (category_x != 'nan' and category_y == 'nan'):
        result = 0
    else:
        if category_x == 'nan' and category_y != 'nan':
            result = 1
        else:
            if category_y in category_x:
                result = 0
            else:
                result = 1
    return result


def is_social_media_domain(domain):
    if pd.isnull(domain):
        return np.nan
    elif isinstance(domain, str):
        for pattern in social_media_patterns:
            if regex.match(pattern, domain):
                return 'Y'
        return 'N'
    else:
        return domain
    
def check_shared_columns(fb_df,g_df,w_df):
    fb_col_dy = set(fb_df.columns)
    g_col_dy = set(g_df.columns)
    w_col_dy = set(w_df.columns)

    common_columns_fb_g = fb_col_dy.intersection(g_col_dy)
    common_columns_fb_w = fb_col_dy.intersection(w_col_dy)
    common_columns_g_w = g_col_dy.intersection(w_col_dy)
    common_columns_all =  fb_col_dy.intersection(g_col_dy).intersection(w_col_dy)

    print(f'Common columns fb-google  : {common_columns_fb_g}')
    print(f'Common columns fb-web     : {common_columns_fb_w}')
    print(f'Common columns google-web : {common_columns_g_w}')
    print(f'Common columns all        : {common_columns_all}')

def check_column_value_counts(df,column):
    print(f'- has the following distribution:\n {df[column].value_counts()}')

def check_null_consistnecy(df, column):
    null_mask = df[column].isnull()
    w_total = len(df)
    w_null_company = len(df[null_mask])
    w_nnull_company = len(df[~null_mask])

    print(f'\t - Null Values    :  {w_null_company} [ {round((w_null_company/w_total)*100,2)} %] \n\t - Not Null Values:  {w_nnull_company} [ {round((w_nnull_company/w_total)*100,2)} %]')


def print_null_analysis(df):
    total_records = len(df)
    null_country  = len(df[df['country_code'].isnull()])
    null_domain  = len(df[df['domain'].isnull()])
    null_city = len(df[df['city'].isnull()])
    null_zip = len(df[df['zip_code'].isnull()])
    null_phones = len(df[df['phone_parsed'].isnull()])
    null_category = len(df[df['category'].isnull()])
    null_address = len(df[df['address'].isnull()])

    print(f'Total records                 : {total_records}    [100.00 %]')
    print(f'Total records no country      :  {null_country}    [ {round((null_country/total_records)*100,2)} %]')
    print(f'Total records no domain       :      {null_domain}    [  {round((null_domain/total_records)*100,2)}  %]')
    print(f'Total records no city         :  {null_city}    [  {round((null_city/total_records)*100,2)} %]')
    print(f'Total records no zip          : {null_zip}    [  {round((null_zip/total_records)*100,2)} %]')
    print(f'Total records no phone_parsed :  {null_phones}    [ {round((null_phones/total_records)*100,2)} %]')
    print(f'Total records no category     :  {null_category}    [ {round((null_category/total_records)*100,2)} %]')
    print(f'Total records no address      :  {null_address}    [ {round((null_address/total_records)*100,2)} %]')

def print_null_analysis_merged(df):
    total_records = len(df)
    null_country  = len(df[df['country_code_x'].isnull()])
    null_domain  = len(df[df['domain'].isnull()])
    null_city = len(df[df['city_x'].isnull()])
    null_zip = len(df[df['zip_code_x'].isnull()])
    null_phones = len(df[df['phone_parsed_x'].isnull()])
    null_category = len(df[df['category_x'].isnull()])
    null_address = len(df[df['address_x'].isnull()])

    print(f'Total records                 : {total_records}    [100.00 %]')
    print(f'Total records no country      :  {null_country}    [ {round((null_country/total_records)*100,2)} %]')
    print(f'Total records no domain       :      {null_domain}    [  {round((null_domain/total_records)*100,2)}  %]')
    print(f'Total records no city         :  {null_city}    [  {round((null_city/total_records)*100,2)} %]')
    print(f'Total records no zip          : {null_zip}    [  {round((null_zip/total_records)*100,2)} %]')
    print(f'Total records no phone_parsed :  {null_phones}    [ {round((null_phones/total_records)*100,2)} %]')
    print(f'Total records no category     :  {null_category}    [ {round((null_category/total_records)*100,2)} %]')
    print(f'Total records no address      :  {null_address}    [ {round((null_address/total_records)*100,2)} %]')


def is_country_code_valid(country_code):
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if country:
            return country.alpha_2
        else:
            return None
    except KeyError:
        return None