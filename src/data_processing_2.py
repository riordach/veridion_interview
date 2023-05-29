
import pandas as pd
import numpy as np
from helpers import checkers, transformers, enrich

full_df = pd.read_parquet('datasets/output/full_df_enriched_3.parquet')
fb_norm_df = pd.read_parquet('datasets/working/fb_norm.parquet')
g_norm_df = pd.read_parquet('datasets/working/g_norm.parquet')
w_norm_df = pd.read_parquet('datasets/working/w_norm.parquet')

mask_websites = full_df['identifier'].str.contains('website')
mask_google = full_df['identifier'].str.contains('google')
mask_facebook = full_df['identifier'].str.contains('facebook')

print('Null Analysys Full Data Frame ')
checkers.print_null_analysis(full_df)

mask_zip_null = full_df['zip_code'].isnull()
countries = full_df[mask_zip_null]['country_code'].value_counts()
mask_zip_null = full_df['zip_code'].isnull() & full_df['address'].notnull()
missing_zip = full_df[mask_zip_null].copy()
missing_zip['zip_code'] = missing_zip.apply(transformers.extract_zip_code, address_column='address', country_code_column='country_code', axis=1)

full_df.update(missing_zip)
print('Null Analysys Full Data Frame [after zip from address]')
checkers.print_null_analysis(full_df)

mask_no_null_on_join = full_df['phone_parsed'].notnull() & full_df['zip_code'].notnull() & full_df['country_code'].notnull()
mask_no_null_on_join_fb = fb_norm_df['phone_parsed'].notnull() & fb_norm_df['zip_code'].notnull() & fb_norm_df['country_code'].notnull()
enrich_fb = full_df[~mask_facebook & mask_no_null_on_join].merge(fb_norm_df[mask_no_null_on_join_fb], how='inner', on=['phone_parsed','zip_code','country_code'])

enrich_fb['name_fuzzy_score'] = enrich_fb.apply(checkers.calculate_company_name_fuzzy_score,column_x='company_name_norm_x', column_y='company_name_norm_y', axis=1)
enrich_fb['category_inclusion_flag'] = enrich_fb.apply(checkers.check_category_inclusion,axis=1)

enrich_fb = enrich.fb_enrichment(enrich_fb,50)

enrich_fb.rename(columns={
    'identifier_x':'identifier',
    'company_name_norm_x': 'company_name_norm',
    'domain_x':'domain',
    'city_x':'city',
    'category_x':'category',
    'address_x':'address',
},inplace=True)

enrich_fb['identifier_index'] = enrich_fb['identifier']

enrich_fb.set_index('identifier_index',inplace=True)

mask_changed_rows =  enrich_fb['category_enriching_id'].notnull() | enrich_fb['domain_enriching_id'].notnull()

normalised_columns = ['identifier','company_name_norm','country_code','phone_parsed','domain','city','zip_code','category','address','social_media_flag','city_enriching_id','country_enriching_id','phone_enriching_id','category_enriching_id','other_company_name','domain_enriching_id']

df = enrich_fb[mask_changed_rows][normalised_columns]
df = df = df[~df.index.duplicated(keep='first')]
full_df['domain_enriching_id'] = None

full_df.update(df)
print('Null Analysys Full Data Frame [after facebook enrichment]')
checkers.print_null_analysis(full_df)

full_df['combined_column'] = full_df[['city_enriching_id','country_enriching_id','phone_enriching_id','category_enriching_id','domain_enriching_id']].apply(lambda x: '|'.join(x.dropna().astype(str)), axis=1)
distinct_df = pd.DataFrame({'unique_values': full_df['combined_column'].str.split('|').explode().unique()})

distinct_df.set_index('unique_values',inplace=True)
full_df_filtered = full_df[~full_df.index.isin(distinct_df.index)]

agg_data = full_df.groupby(['company_name_norm','country_code','domain'])\
                     .apply(lambda df: pd.Series({
                                          'rows_count': len(df),
                                          'categories': '|'.join(df['category'].dropna().unique()),
                                          'addresses': transformers.create_json_address(df)
                                                        })).reset_index()

agg_data.to_parquet('datasets/output/final_agg_data.parquet')