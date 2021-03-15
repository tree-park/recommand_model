import pandas as pd
pd.set_option('display.max_columns', 500)

data = pd.read_csv('../data/bungae_test/rec-exam.csv000.gz',
                   compression='gzip',
                   quotechar='"',
                   escapechar='\\',
                   dtype=str,
                   nrows=100)
print(data[['content_type', 'ref_term', 'name', 'keyword', 'category_name']].head(10))