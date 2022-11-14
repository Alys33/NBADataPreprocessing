import pandas as pd
import os
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"
def clean_data(path):

# Stage 1/4
    df = pd.read_csv(path)

    df['b_day'] = pd.to_datetime(df["b_day"])
    df['draft_year'] = pd.to_datetime(df["draft_year"], format='%Y')
    df["team"].fillna("No Team", inplace=True)

    df[["false_height","height"]]= df["height"].str.split("/", expand=True)
    df["height"] = df["height"].astype("float")

    df.drop(columns = "false_height", inplace=True)

    df[["false_weig","weight"]] = df["weight"].str.split("/", expand=True)
    df.drop(columns ="false_weig", inplace=True)
    df["weight"] = df["weight"].str.replace("kg.", "").astype("float")
    df["salary"] = df["salary"].str.replace("$", "").astype("float")
    df["country"] = df["country"].apply(lambda x: "Not-USA" if x != "USA" else "USA")
    df["draft_round"] = df["draft_round"].str.replace("Undrafted", '0')
    return df


df_cleaned = clean_data(data_path)

# Stage 2/4
def feature_data(df):
    df["version"] = df["version"].apply(lambda x: "12-31-2020" if x == "NBA2k20" else "12-31-2021")
    df["version"] = pd.to_datetime(df["version"])

    df["age"] = (df["version"] - df["b_day"]).astype("<m8[Y]")
    df["age"] = df["age"].astype("int")

    df["experience"] = (df["version"] - df["draft_year"]) / np.timedelta64(1, "Y")
    df["experience"] = df["experience"].astype("int")
    columns = ["age","experience","bmi"]

    df["bmi"] = df["weight"] / (df["height"] ** 2)

    df.drop(columns = ["version", "b_day","draft_year", "weight","height"], inplace=True)
    for el in ['full_name', 'rating', 'jersey', 'team', 'position','country', 'draft_round', 'draft_peak', 'college']:
        if df[el].nunique() >=50:
            df = df.drop(columns=el)
    return df


df_featured = feature_data(df_cleaned)
# print(df[["age","experience","bmi"]].head())


# Stage 3/4


def multicol_data(df):
    new_df = df[["rating","bmi","experience","age"]]
    corr = new_df.corr()
    my_list = []
    corr_values = []
    columns = ['rating','bmi','experience','age']
    for el in columns:
        for n_el in columns:
            if el != n_el:
                new_corr = df[[el, n_el]].corr()
                if np.logical_or(new_corr[el][n_el] > 0.5, new_corr[el][n_el] < -0.5):
                    my_list.append(el)
                    my_list.append(n_el)

    for el in my_list:
        corr2 = df[["salary",el]].corr()
        corr_values.append(corr2["salary"][el])
    index = corr_values.index(min(corr_values))
    df.drop(columns=my_list[index], inplace=True)
    return df




df = multicol_data(df_featured)

#print(df.columns)
#print(list(df.select_dtypes('number').drop(columns='salary')))

# Stage 4/4: Transform to high quality

def transform_data(df):
    num_feat_df = df.select_dtypes('number') # for numerical features + the target feature
    X_cat = df.select_dtypes('object') # for categorical features
    X_num = num_feat_df[['rating','experience', 'bmi']]
    y = num_feat_df['salary']
    # transforming numerical feature
    scaler = StandardScaler()
    scaler.fit(X_num)
    X_num_tr = scaler.transform(X_num)
    X_num_tr2 = pd.DataFrame({'rating': X_num_tr[:, 0],
                              'experience': X_num_tr[:, 1],
                              'bmi': X_num_tr[:, 2]})

    # transforming categorical feature
    encoder = OneHotEncoder()
    encoder.fit(X_cat)  # to access the features' names, use encoder.categories
    categories = list(encoder.categories_[0]) + list(encoder.categories_[1]) + list(encoder.categories_[2]) + list(encoder.categories_[3])
    X_cat_sparse = encoder.transform(X_cat)
    X_cat_tran = pd.DataFrame.sparse.from_spmatrix(X_cat_sparse, columns=categories)


    # concatenate the two features to obtain the whole feature set
    X = pd.concat([X_num_tr2, X_cat_tran], axis=1)

    return X, y



#
X, y = transform_data(df)
answer = {
    "shape": [X.shape, y.shape],
    'features': list(X.columns)
}

print(answer)




