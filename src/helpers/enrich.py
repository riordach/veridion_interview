import numpy as np


def city_enrichment(df, ratio):
    #City Enrichment from Websites
    df['city_enriched'] = np.where(
        (df['city_x'].isnull() & df['city_y'].notnull())
        & (
        (df['company_name_norm_x'] == df['company_name_norm_y'])
        | ((df['name_fuzzy_score']>ratio) & (df['city_y'].notnull()))
        | ((df['phone_parsed_x'] == df['phone_parsed_y']) & df['phone_parsed_x'].notnull())
        | ((df['company_name_norm_x'] == df['company_name_norm_y'] )& (df['country_code_x'] == df['country_code_y']) & df['country_code_x'].notnull())
        ),
        df['city_y'],
        None
    )

    df['city_enriching_id'] = np.where(
        (df['city_x'].isnull() & df['city_y'].notnull())
        & (
        (df['company_name_norm_x'] == df['company_name_norm_y'])
        | ((df['name_fuzzy_score']>ratio) & (df['city_y'].notnull()))
        | ((df['phone_parsed_x'] == df['phone_parsed_y']) & df['phone_parsed_x'].notnull())
        | ((df['company_name_norm_x'] == df['company_name_norm_y'] )& (df['country_code_x'] == df['country_code_y']) & df['country_code_x'].notnull())
        ),
        df['identifier_y'],
        None
    )

    total_records = len(df)
    # null_country  = len(df[df['country_code_x'].isnull()])
    # null_domain  = len(df[df['domain'].isnull()])
    null_city = len(df[df['city_x'].isnull()])
    # null_zip = len(df[df['zip_code_x'].isnull()])
    # null_phones = len(df[df['phone_parsed_x'].isnull()])
    # null_category = len(df[df['category_x'].isnull()])
    # null_address = len(df[df['address_x'].isnull()])
    total_enriched = len(df[df['city_enriching_id'].notnull()])
    print(f'Total enriched cities : {total_enriched} [ {round((total_enriched/null_city)*100,2)} %] [ {round((total_enriched/total_records)*100,2)} %]')

    return df

def country_enrich(df,ratio):
    #Country Enrichment from Websites
    df['country_enriched'] = np.where( 
        ((df['country_code_x'].isnull())) 
        &(
        ((df['name_fuzzy_score']>ratio) & (df['country_code_y'].notnull()))
        | ((df['phone_parsed_x'] == df['phone_parsed_y']) & df['phone_parsed_x'].notnull())
        ),
        df['country_code_y'],
        None
    )

    df['country_enriching_id'] = np.where(
        ((df['country_code_x'].isnull()))
        &(
        ((df['name_fuzzy_score']>ratio) & (df['country_code_y'].notnull()))
        | ((df['phone_parsed_x'] == df['phone_parsed_y']) & df['phone_parsed_x'].notnull())
        ),
        df['identifier_y'],
        None
    )
    total_records = len(df)
    null_country  = len(df[df['country_code_x'].isnull()])
    total_enriched_country = len(df[df['country_enriching_id'].notnull()])
    print(f'Total enriched countries : {total_enriched_country} [ {round((total_enriched_country/null_country)*100,2)} %] [ {round((total_enriched_country/total_records)*100,2)} %]')

    return df

def phone_enrichement(df, ratio):
    #Phone Enrichment from Websites
    df['phone_enriched'] = np.where( 
        ((df['phone_parsed_x'].isnull()) | (df['phone_parsed_x']=='nan') ) 
        &(
        ((df['name_fuzzy_score']>ratio) & (df['phone_parsed_y'].notnull()))
        ),
        df['phone_parsed_y'],
        None
    )

    df['phone_enriching_id'] = np.where(
        ((df['phone_parsed_x'].isnull()) | (df['phone_parsed_x']=='nan') ) 
        &(
        ((df['name_fuzzy_score']>ratio) & (df['phone_parsed_y'].notnull()))
        ),
        df['identifier_y'],
        None
    )
    total_records = len(df)
    null_phones = len(df[df['phone_parsed_x'].isnull()])

    total_enriched_phones = len(df[df['phone_enriching_id'].notnull()])
    print(f'Total enriched phones : {total_enriched_phones} [ {round((total_enriched_phones/null_phones)*100,2)} %] [ {round((total_enriched_phones/total_records)*100,2)} %]')

    return df

def category_enrichment(df, ratio):
    #Category Enrichment from Websites
    df['category_enriched'] = np.where( 
        ((df['category_inclusion_flag']>0) & (df['category_x'].notnull())) 
        &(
        ((df['name_fuzzy_score']>ratio) & (df['category_y'].notnull()))
        | ((df['phone_parsed_x'] == df['phone_parsed_y']) & df['phone_parsed_x'].notnull())
        ),
        df['category_x'] + '|' + df['category_y'],
        np.where(
            ((df['category_inclusion_flag']>0) & (df['category_x'].isnull())) 
        &(
        ((df['name_fuzzy_score']>ratio) & (df['category_y'].notnull()))
        | ((df['phone_parsed_x'] == df['phone_parsed_y']) & df['phone_parsed_x'].notnull())
        ),
        df['category_y'],
        None
        )
    )

    df['category_enriching_id'] = np.where(
        ((df['category_inclusion_flag']>0)) 
        &(
        ((df['name_fuzzy_score']>ratio) & (df['category_y'].notnull()))
        | ((df['phone_parsed_x'] == df['phone_parsed_y']) & df['phone_parsed_x'].notnull())
        ),
        df['identifier_y'],
        None
    )
    total_records = len(df)
    null_category = len(df[df['category_x'].isnull()])
    total_enriched_categories = len(df[df['category_enriching_id'].notnull()])
    print(f'Total enriched categories : {total_enriched_categories} [ {round((total_enriched_categories/null_category)*100,2)} %] [ {round((total_enriched_categories/total_records)*100,2)} %]')

    return df

def fb_enrichment(enrich_fb, ratio):

    enrich_fb['domain_x'] = np.where(((enrich_fb['social_media_flag']=='Y') & (enrich_fb['name_fuzzy_score']>ratio)), enrich_fb['domain_y'], enrich_fb['domain_x'])
    enrich_fb['domain_enriching_id'] = np.where(((enrich_fb['social_media_flag']=='Y') & (enrich_fb['name_fuzzy_score']>ratio)), enrich_fb['identifier_y'], None)

    #Category Enrichment from Websites
    enrich_fb['category_x'] = np.where( 
        ((enrich_fb['category_inclusion_flag']>0) & (enrich_fb['category_x'].notnull())) 
        &(
        ((enrich_fb['name_fuzzy_score']>ratio) & (enrich_fb['category_y'].notnull()))
        ),
        enrich_fb['category_x'] + '|' + enrich_fb['category_y'],
        np.where(
            ((enrich_fb['category_inclusion_flag']>0) & (enrich_fb['category_x'].isnull())) 
        &(
        ((enrich_fb['name_fuzzy_score']>50) & (enrich_fb['category_y'].notnull()))
        ),
        enrich_fb['category_y'],
    None
        )
    )

    enrich_fb['category_enriching_id'] = np.where(
        ((enrich_fb['category_inclusion_flag']>0) & (enrich_fb['category_enriching_id'].isnull())) 
        &(
        ((enrich_fb['name_fuzzy_score']>ratio) & (enrich_fb['category_y'].notnull()))
        ),
        enrich_fb['identifier_y'],
    np.where(
            ((enrich_fb['category_inclusion_flag']>0) & (enrich_fb['category_enriching_id'].notnull())) 
        &(
        ((enrich_fb['name_fuzzy_score']>ratio) & (enrich_fb['category_y'].notnull()))
        ),
        enrich_fb['category_enriching_id']+ "|" + enrich_fb['identifier_y'],
        None
    )
    )
    return enrich_fb