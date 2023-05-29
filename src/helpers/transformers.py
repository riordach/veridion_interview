import phonenumbers as ph
import pycountry as pc
import pandas as pd
import numpy as np
import regex
import json

def format_and_validate_phone_number(phone_number):
    try:
        if pd.isna(phone_number):
            phone_number = ''  # Return an empty string for NaN values
        elif not str(phone_number).startswith('+'):
            # Handle non-NaN values that don't start with '+'
            # Your desired logic here, e.g., formatting or validation
            phone_number = '+' + phone_number
        else:
            # Handle non-NaN values that already start with '+'
            # Your desired logic here, e.g., formatting or validation
            phone_number = phone_number
        parsed_number = ph.parse(phone_number, None)
        formatted_number = ph.format_number(parsed_number, ph.PhoneNumberFormat.INTERNATIONAL)
        return formatted_number if ph.is_valid_number(parsed_number) else None
    except ph.phonenumberutil.NumberParseException:
        return None
    
def parse_country_name_to_code(country_name):
    try:
        country = pc.countries.get(name=country_name)
        if country is not None:
            return str.lower(country.alpha_2)
        else:
            return None
    except LookupError:
        return None
    
def normalise_comapny_name(company_name):
    if pd.isna(company_name) or not isinstance(company_name, str):
        return 'None'  # Return an empty string for NaN or non-string values
    else:
        # Define a list of company types to remove
        company_types = ['Ltd', 'Inc', 'Corp', 'LLC', 'Ltd.', 'Inc.', 'Corp.', 'LLC.', 'Co', 'AG', 'PLC', 'SA', 'SRL', 'NV', 'Pty', 'AB', 'BV', 'MD']

    # Remove special characters from the company name
        company_name = regex.sub(r'\p{P}', '', company_name) #removing special characters
        company_name = regex.sub(r'\+[0-9]+|\<|\>|°|^[0-9]+$', '',company_name) # removingspecial characters not removed previously
        company_name = regex.sub(r'[^\w\s]', '', company_name) # removing emoticons

        # Use regular expression to match and remove company types
        pattern = regex.compile(r'\b(?:{})\b'.format('|'.join(map(regex.escape, company_types))), regex.IGNORECASE)
        company_name = pattern.sub('', company_name)

        # Remove leading/trailing whitespaces and reduce multiple spaces to a single space
        company_name = regex.sub(r'\s+', ' ', company_name).strip()
        if (str.upper(company_name) == 'NAN' or company_name == ''):
            return None
        else:
            return str.upper(company_name)
    
def clean_city(string):
    if pd.isnull(string):
        return None
    elif isinstance(string, str):
        string = regex.sub( r"[^\w\s]", '', string)  # Remove special characters
        string = string.strip()  # Remove trailing and leading spaces
        return string
    else:
        return string
    
def create_json_address(df):
    json_addresses = []
    unique_addresses = set()
    for _, row in df.iterrows():
        address = {
            'id': row['identifier'],
            'phone': row['phone_parsed'],
            # 'country': row['country_code'],
            'city': row['city'],
            'zip_code': row['zip_code'],
            'raw_address': row['address']
        }
        address_str = json.dumps(address)
        if address_str not in unique_addresses:
            json_addresses.append(address)
            unique_addresses.add(address_str)
        
    return json.dumps(json_addresses)

def create_normalised_dataframe(df,columns):
    """
    This function filters the inconsistent domains as well.
    """
    df_norm = df[df['valid_domain']==0][columns].copy()
    df_norm = df_norm.replace(['NaN','nan','NAN'], None)

    return df_norm

def extract_zip_code(row, address_column, country_code_column):
    country_code = str(row[country_code_column])
    text = str(row[address_column])
    pattern = ""

    if country_code == "us":
        pattern = r"(?i)\b\d{5}(?:-\d{4})?\b"
    elif country_code == "ca":
        pattern = r"(?i)\b[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d\b"
    elif country_code == "gb":
        pattern = r"(?i)\b(?:[A-Z]{1,2}\d[A-Z\d]?|\d[A-Z\d]{1,2}) \d[A-Z]{2}\b"
    elif country_code == "uk":
        pattern = r"(?i)\b(?:[A-Z]{1,2}\d[A-Z\d]?|\d[A-Z\d]{1,2}) \d[A-Z]{2}\b"
    elif country_code == "au":
        pattern = r"(?i)\b\d{4}\b"
    elif country_code == "tr":
        pattern = r"(?i)\b\d{5}\b"
    else:
        return None

    match = regex.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return None
    
def extract_tdl(domain):
    pattern = r"(.*\.)(.*$)"
    match = regex.search(pattern, domain)
    if match:
        return match.group(2)
    else:
        return None
    
def get_country_from_phone(phone_number):
    try:
        parsed_number = ph.parse(phone_number, None)
        country_code = ph.region_code_for_number(parsed_number)
        # country_name = phonenumbers.region_name_for_number(parsed_number)
        return country_code.lower()
    except ph.phonenumberutil.NumberParseException:
        return None