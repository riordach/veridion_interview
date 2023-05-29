import pandas as pd
import numpy as np
from helpers import checkers, transformers, enrich


#Load data into dataframes
#Reading data into pandas dataframes
fb_df = pd.read_csv('datasets/facebook_dataset.csv', delimiter=",", quotechar='"', escapechar='\\' ,dtype=str)
g_df = pd.read_csv('datasets/google_dataset.csv', delimiter=",", quotechar='"', escapechar='\\' ,dtype=str)
w_df = pd.read_csv('datasets/website_dataset.csv', delimiter=";", quotechar='"',dtype=str)

#Filling in null values
fb_df = fb_df.replace(['NaN','nan','NAN',''], None)
g_df = g_df.replace(['NaN','nan','NAN',''], None)
w_df = w_df.replace(['NaN','nan','NAN',''], None)

#Creating an identifier column to let-us know the source of the data
fb_df['identifier'] = fb_df.index.map(lambda x: str(x) + '-facebook') 
g_df['identifier'] = g_df.index.map(lambda x: str(x) + '-google')
w_df['identifier'] = w_df.index.map(lambda x: str(x) + '-website')

print('Common columns of the datasets:')
checkers.check_shared_columns(fb_df,g_df,w_df)

#Renaming columns to be the same where there is same content
w_df.rename(columns={'root_domain':'domain',
                    'main_city':'city',
                    'main_country':'country_name',
                    'main_region':'region',
                    's_category':'category'}, inplace=True)

fb_df.rename(columns={'categories':'category'
                    }, inplace=True)
print(f'Common columns after renaming:')
checkers.check_shared_columns(fb_df,g_df,w_df)

#Checking domain colums to see if it contains a valid domain string
# Regex pattern for domain validation
regex_pattern = r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

fb_df['valid_domain'] = np.where(fb_df['domain'].str.match(regex_pattern),0,1) # 0 - true, 1 - false
g_df['valid_domain'] = np.where(g_df['domain'].str.match(regex_pattern),0,1) # 0 - true, 1 - false
w_df['valid_domain'] = np.where(w_df['domain'].str.match(regex_pattern),0,1) # 0 - true, 1 - false

print('Facebook DF domain data quality:')
print(fb_df['valid_domain'].value_counts())
print('Google DF domain data quality:')
print(g_df['valid_domain'].value_counts())
print('Websites DF domain data quality:')
print(w_df['valid_domain'].value_counts())

# Checking and parsing phone data
fb_df['phone_parsed'] = fb_df['phone'].apply(transformers.format_and_validate_phone_number)
g_df['phone_parsed'] = g_df['phone'].apply(transformers.format_and_validate_phone_number)
w_df['phone_parsed'] = w_df['phone'].apply(transformers.format_and_validate_phone_number)

#Normalising Company Names
fb_df['company_name_norm'] = fb_df['name'].apply(transformers.normalise_comapny_name)
g_df['company_name_norm'] = g_df['name'].apply(transformers.normalise_comapny_name)
w_df['company_name_norm'] = w_df['legal_name'].apply(transformers.normalise_comapny_name)

#getting country code for websites df
w_df['country_code'] = w_df['country_name'].apply(transformers.parse_country_name_to_code)

#adding missing columns with null for the websites df
w_df['address'] = None
w_df['zip_code'] = None

checkers.check_shared_columns(fb_df,g_df,w_df)

normalised_columns = ['identifier','company_name_norm','country_code','phone_parsed','domain','city','zip_code','category','address']
fb_norm_df = transformers.create_normalised_dataframe(fb_df,normalised_columns)
g_norm_df = transformers.create_normalised_dataframe(g_df,normalised_columns)
w_norm_df = transformers.create_normalised_dataframe(w_df,normalised_columns)

fb_norm_df.to_parquet('datasets/working/fb_norm.parquet')
g_norm_df.to_parquet('datasets/working/g_norm.parquet')
w_norm_df.to_parquet('datasets/working/w_norm.parquet')

## Phase 2 - Creating full dataset - the output one
print('Google dataframe domain:')
checkers.check_column_value_counts(g_norm_df,'domain')
print('Websites company name consistency: \n')
checkers.check_null_consistnecy(w_norm_df,'company_name_norm')
print('Facebook company name consistency: \n')
checkers.check_null_consistnecy(fb_norm_df,'company_name_norm')
print('Google company name consistency: \n')
checkers.check_null_consistnecy(g_norm_df,'company_name_norm')

#Getting all companies we have a name for
c_fb_norm_df = fb_norm_df[fb_norm_df['company_name_norm'].notnull()].copy()
c_fb_norm_df = c_fb_norm_df.reset_index(drop=True)

c_g_norm_df = g_norm_df[g_norm_df['company_name_norm'].notnull()].copy()
c_g_norm_df = c_g_norm_df.reset_index(drop=True)

c_w_norm_df = w_norm_df[w_norm_df['company_name_norm'].notnull()].copy()
c_w_norm_df = c_w_norm_df.reset_index(drop=True)

full_df = pd.concat( [c_fb_norm_df, c_g_norm_df ,c_w_norm_df], ignore_index=True)
full_df = full_df.replace(['NaN','nan','NAN',''],None)
full_df.shape

#checking city for special characters
full_df['city'] = full_df['city'].apply(transformers.clean_city)
full_df = full_df.replace(['NaN','nan','NAN',''],None)
full_df.to_parquet('datasets/working/full_df.parquet')
print('Null Analysys Full Data Frame [before any enrichment]')
checkers.print_null_analysis(full_df)

# Phase 3 - Enriching full data with websites information

google_domains = g_norm_df['domain'].value_counts()
facebook_domains = fb_norm_df['domain'].value_counts()
websites_domains = w_norm_df['domain'].value_counts()

full_df['social_media_flag'] = full_df['domain'].apply(checkers.is_social_media_domain)
mask_no_social_media = (full_df['social_media_flag'] == 'N')

g_soc_med = full_df[~mask_no_social_media]

full_df_no_soc = full_df[mask_no_social_media]

mask_websites = full_df['identifier'].str.contains('website')
mask_google = full_df['identifier'].str.contains('google')
mask_facebook = full_df['identifier'].str.contains('facebook')

#Getting full_df wihtout websites to enrich it furhter with data in websites

no_w = full_df[~mask_websites].merge(w_norm_df, how='left', on=['domain']) #merging on domain as there is a lot of consistency and low missing values
no_w.reset_index(drop=True, inplace=True)

no_w['name_fuzzy_score'] = no_w.apply(checkers.calculate_company_name_fuzzy_score, column_x='company_name_norm_x', column_y='company_name_norm_y', axis=1)
no_w['category_inclusion_flag'] = no_w.apply(checkers.check_category_inclusion,axis=1)
print('Null Analysis Full Data Frame no websites')
checkers.print_null_analysis_merged(no_w)
print('Enriching google and facebookd data with websites data')
#City Enrichment from Websites
no_w = enrich.city_enrichment(no_w,50)
#Country Enrichment from Websites
no_w = enrich.country_enrich(no_w,50)

#Phone Enrichment from Websites
no_w = enrich.phone_enrichement(no_w,50)

#Category Enrichment from Websites
no_w = enrich.category_enrichment(no_w,50)

no_w.to_parquet('datasets/working/no_web.parquet')
df = pd.read_parquet('datasets/working/no_web.parquet')

#Generating the final enriched dataset.
mask_country = df['country_enriching_id'].notnull() & df['country_code_x'].isnull()
mask_phone = df['phone_enriching_id'].notnull() & df['phone_parsed_x'].isnull()
mask_city = df['city_enriching_id'].notnull() & df['city_x'].isnull()
mask_category = df['category_enriching_id'].notnull() & df['category_x'].isnull()

df.loc[mask_country, 'country_code_x'] = df.loc[mask_country, 'country_enriched'] #14350
df.loc[mask_phone, 'phone_parsed_x'] = df.loc[mask_phone, 'phone_enriched'] #14350
df.loc[mask_city, 'city_x'] = df.loc[mask_city, 'city_enriched'] #14350
df.loc[mask_category, 'category_x'] = df.loc[mask_category, 'category_enriched'] #14350

print('Null Analysis Full Data Frame')
checkers.print_null_analysis_merged(df)
#Generating the dataframe to update the full_df
#'city_enriching_id','country_enriching_id','phone_enriching_id','category_enriching_id'
mask_changed_rows = df['city_enriching_id'].notnull() | df['country_enriching_id'].notnull() | df['phone_enriching_id'].notnull() | df['category_enriching_id'].notnull()
normalised_columns = ['identifier','company_name_norm','other_company_name','country_code','phone_parsed','domain','city','zip_code','category','address','social_media_flag','city_enriching_id','country_enriching_id','phone_enriching_id','category_enriching_id']
df.rename(columns={
    'identifier_x':'identifier',
    'company_name_norm_x': 'company_name_norm',
    'country_code_x': 'country_code',
    'phone_parsed_x': 'phone_parsed',
    'city_x':'city',
    'zip_code_x':'zip_code',
    'category_x':'category',
    'address_x':'address',
    'company_name_norm_y':'other_company_name',
},inplace=True)
no_w_enriched = df[mask_changed_rows][normalised_columns]

full_df['city_enriching_id'] = None
full_df['country_enriching_id'] = None
full_df['phone_enriching_id'] = None
full_df['category_enriching_id'] = None
full_df['other_company_name'] = None

full_df['identifier_index'] = full_df['identifier']
full_df.set_index('identifier_index', inplace=True)

no_w_enriched['identifier_index'] = no_w_enriched['identifier']
no_w_enriched.set_index('identifier_index', inplace=True)
#Updating the full_df with the fixed data
full_df.update(no_w_enriched)

full_df.to_parquet('datasets/output/full_df_web_enriched.parquet')

print('Null Analysys Full Data Frame [after websites enrichment]')
checkers.print_null_analysis(full_df)

mask_phone_no_country = ((full_df['phone_parsed'].notnull()) & (full_df['country_code'].isnull()))

df = full_df[mask_phone_no_country].copy()

df['country_code']=df['phone_parsed'].apply(transformers.get_country_from_phone)
df['country_code'] = df['country_code'].str.lower()

full_df.update(df)

print('Null Analysys Full Data Frame [after country from phone]')
checkers.print_null_analysis(full_df)

mask_address_not_null = ((full_df['country_code'].isnull()) & (full_df['social_media_flag'] == 'N'))

missing_address = full_df[mask_address_not_null].copy()

missing_address['tdl'] = missing_address['domain'].apply(transformers.extract_tdl)
missing_address['is_tdl_country'] = missing_address['tdl'].apply(checkers.is_country_code_valid)
missing_address['country_code'] = np.where(missing_address['is_tdl_country'].notnull(), missing_address['tdl'],None)
missing_address.drop(columns=['tdl','is_tdl_country'], axis=1, inplace=True)

full_df.update(missing_address)

print('Null Analysys Full Data Frame [after country from domain]')
checkers.print_null_analysis(full_df)
full_df.to_parquet('datasets/output/full_df_enriched_3.parquet')


mask_zip_null = full_df['zip_code'].isnull()
countries = full_df[mask_zip_null]['country_code'].value_counts()
mask_zip_null = full_df['zip_code'].isnull() & full_df['address'].notnull()
missing_zip = full_df[mask_zip_null].copy()
missing_zip['extracted_zip'] = missing_zip.apply(transformers.extract_zip_code, address_column='address', country_code_column='country_code', axis=1)

print('Null Analysys Full Data Frame [after zip from address]')
checkers.print_null_analysis(full_df)

