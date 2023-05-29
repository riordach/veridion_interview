import pandas as pd
from fuzzywuzzy import fuzz
from multiprocessing import Pool

google = pd.read_parquet('datasets/working/g_norm.parquet')
google = google.astype(str)

df1 = google.head(1000)

# pd.DataFrame({
#     'identifier': ['1-src','2-src','3-src','4-src','5-src'],
#     'company_name': ['Apple','Google', 'Chimacum School Distrinct', 'Autism Partnership', 'Autism Hospital'],
#     'country': ['us', None, 'ca', None, None]
# })

facebook = pd.read_parquet('datasets/working/fb_norm.parquet')
facebook = facebook.astype(str)
df2 = facebook
# pd.DataFrame({
#     'identifier': ['1-arc','2-arc'],
#     'company_name': ['Apple Inc', 'Chimacum'],
#     'country': ['us', None]
# })

def fuzzy_match(row):
    best_match = None
    score = 0

    for index, df2_row in df2.iterrows():
        current_score = fuzz.token_set_ratio(row['company_name_norm'], df2_row['company_name_norm'])
        if current_score > score:
            score = current_score
            best_match = df2_row['company_name_norm']

    return row['identifier'], row['company_name_norm'], best_match, score

def process_row(row):
    return fuzzy_match(row[1])

def main():
    matched = pd.DataFrame(columns=['identifier', 'company_name_norm', 'best_match', 'score'])
    total_rows = len(df1)
    processed_rows = 0

    with Pool(10) as p:
        results = p.map(process_row, df1.iterrows())

    for result in results:
        if result[3] > 50:  # Filter matches with score > 50
            matched = matched.append(pd.Series(result, index=matched.columns), ignore_index=True)
        processed_rows += 1
        progress = (processed_rows / total_rows) * 100
        print(f"Processing progress: {progress:.2f}%")

    matched.to_parquet('datasets/working/merged_G_fb.parquet')
    print(matched)

if __name__ == '__main__':
    main()
