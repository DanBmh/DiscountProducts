import pandas as pd
import numpy as np

categories = pd.read_csv('data/category_names.csv')

super_cats = categories['category_level1'].unique().tolist()
parent_cats = categories['category_level2'].unique().tolist()
normal_cats = categories['category_id'].unique().tolist()


def toNumber(val, where):
    return where.index(val)


categories['category_level1'] = categories['category_level1'].apply(
    toNumber, args=(super_cats,))
categories['category_level2'] = categories['category_level2'].apply(
    toNumber, args=(parent_cats,))
categories['category_level3'] = categories['category_id'].apply(
    toNumber, args=(normal_cats,))

categories.to_csv('data/category_numbers.csv', index=False)
