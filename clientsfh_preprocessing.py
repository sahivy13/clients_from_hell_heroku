import pandas as pd
import numpy as np
import regex as re

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


import streamlit as st

from sqlalchemy import create_engine
import psycopg2

# import pickle
# import os
# from sklearn.preprocessing import MinMaxScaler

# --- Stating Random Seed
np.random.seed(42)

def df_creator(dic_):
    df_ = pd.DataFrame.from_dict(dic_, orient = 'index').fillna('').transpose()
    return df_

def stem(sentence : string):
    p = PorterStemmer()
    sentence = [p.stem(word) for word in sentence]
    return sentence

def cleaning(df : pd.DataFrame):
    
    for col in df:

        for i,list_ in enumerate(df[col]):
            
            sub_list=[]

            for item in list_:
                if item.startswith('Client:'):
                    sub_list.append(item)

            df[col][i] = sub_list
    

    punc_list = [x for x in string.punctuation]

    for col in df:

        for i,list_ in enumerate(df[col]):

            sub_list = [x.replace('\xa0|\n|Client: ', ' ') for x in df[col][i]]
            
            for punc in punc_list:
                sub_list = [x.replace(punc, '') for x in sub_list]
                
            sub_list = [x.replace('—|   |  ', '').rstrip() for x in sub_list]

            df[col][i] = sub_list
            

    for col in df:

        for i,list_ in enumerate(df[col]):

            sub_list = [x.split(' ') for x in list_]

            df[col][i] = sub_list
            df[col][i] = [word.lower() for words in df[col][i] for word in words if len(word) != 1]
            df[col][i] = [re.sub(r'^(.)\1+', r'\1', word)  for word in df[col][i]]
            df[col][i] = [word.replace("’", "'") for word in df[col][i]]
            df[col][i] = [word.replace("client", "") for word in df[col][i]]
            df[col][i] = [word.rstrip("'") for word in df[col][i]]

            df[col][i] = [word for word in df[col][i] if word not in stopwords.words('english')]
            df[col][i] = [word for word in df[col][i] if word.isalpha() == True]
            df[col][i] = [word for word in df[col][i] if len(word) != 1]
            df[col][i] = stem(df[col][i])

    
    df_final = df.transpose()

    df_final.columns = [str(col) for col in df_final.columns]

    df_final.reset_index(inplace = True)
    df_final.rename(columns = {'index':'category'}, inplace = True)

    df_cases = pd.DataFrame(columns = ['category', 'case'])

    for col in df_final:
        if col != 'category':
            df_cases = df_cases.append(df_final[['category', col]].rename(columns = {col:'case'}))

    df_cases.reset_index(drop = True, inplace = True)

    for i, row in enumerate(df_cases['case']):
        if row == []:
            df_cases.drop(index = i, inplace = True)

    df_cases['case'] = df_cases['case'].apply(lambda x: ' '.join(x))
    df_cases.reset_index(drop = True, inplace = True) #ADDED

    return df_cases

def category_replacer(df, col = 'category', mul = True, main_cat = "Deadbeats"):

    if mul == True: #--- MULTILABEL ---
        dic_cat = {}
        for i, cat in enumerate(list(df[col].unique())):
            dic_cat[cat] = i

    else: #--- CATEGORY VS. NOT CATEGORY ---
        dic_cat = {
            "Deadbeats" : 0,
            'Dunces' : 0,
            'Criminals' : 0,
            'Racists' : 0,
            'Homophobes' : 0,
            'Sexist' : 0,
            'Frenemies' : 0,
            'Cryptic' : 0,
            'Ingrates' : 0,
            'Chaotic Good' : 0
        }
        
        dic_cat[main_cat]  =  1


    def cat_to_db(df_):
        
        param_dic = {
        "user" : "onpsmhcjnzdsiz",
        "password" : "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
        "host" : "ec2-3-216-92-193.compute-1.amazonaws.com",
        "port" : "5432", #Postgres Port
        "database" : "dc1fq03u49u20u"
        }

        connect = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
            param_dic['user'],
            param_dic['password'],
            param_dic['host'],
            param_dic['port'],
            param_dic['database']
        )

        engine = create_engine(connect)

        df_.to_sql(
            'category_labels', 
            con=engine, 
            index=False, 
            if_exists='replace'
        )

    df_cat_dict = pd.DataFrame(dic_cat, index=[0])
    cat_to_db(df_cat_dict)

    df[col].replace(to_replace = dic_cat, inplace = True)
    
    return df

def over_under_sampling(df):
    n_y_num_samples = st.sidebar.selectbox("Manually choose number of samples?", ("No", "Yes"))

    list_count = Counter(df['category']).most_common()
    mid_num = int(((list_count[0][1]-list_count[9][1])/2))
    mid_num = mid_num+list_count[9][1]

    if n_y_num_samples == "Yes":
        num_samples = st.sidebar.slider(
            "Choose # of samples for each category:",
            min_value=list_count[9][1],
            max_value=list_count[0][1],
            value=mid_num
        )
        if st.sidebar.button("Re-Train"):
            # for path in ['bernoulli.pickle', 'guassian.pickle', 'knn.pickle', 'log_regr.pickle', 'multi.pickle', 'rfc.pickle']:
            #     if os.path.exists(path):
            #         os.remove(path)
            st.caching.clear_cache()
    else:
        num_samples = mid_num
        if st.sidebar.button("Re-Train"):
            # for path in ['bernoulli.pickle', 'guassian.pickle', 'knn.pickle', 'log_regr.pickle', 'multi.pickle', 'rfc.pickle']:
            #     if os.path.exists(path):
            #         os.remove(path)
            st.caching.clear_cache()

    strategy = dict(Counter(df['category']))
    
    for i in range(len(strategy)):
        if strategy[i] < num_samples:
            strategy[i] = num_samples
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=42)
    rus = RandomUnderSampler(random_state=42)

    X = df.drop(['category'], axis=1)
    y = df[['category']]

    X_ros, y_ros = ros.fit_resample(X, y)
    X_rous, y_rous = rus.fit_resample(X_ros, y_ros)
    o_u_sample = X_rous.join(y_rous)

    return o_u_sample 

def under_sampling_to_2_col_by_index(df, high = 0, low = 1, col_name = 'category'):
    
    low_size = len(df[df[col_name] == low])
    high_indices = df[df[col_name] == high].index
    # mid_indices = df[df[col_name] == mid].index
    
    low_indices = df[df[col_name] == low].index
    random_high_indices = np.random.choice(high_indices, low_size, replace=False)
    # random_mid_indices = np.random.choice(mid_indices, low_size, replace=False)
    
    under_sample_indices = np.concatenate([random_high_indices,low_indices]) #,random_mid_indices
    
    under_sample = df.loc[under_sample_indices]
    under_sample = under_sample.reset_index(drop = True)
    
    return under_sample

def convert_to_tfidf(df, case_col = 'case', target_col = 'category'):
    
    tfidf = TfidfVectorizer()
    word_count_vectors = tfidf.fit_transform(df[case_col].values).todense().tolist()
    
    features = pd.DataFrame(
    data = word_count_vectors,
    columns = tfidf.get_feature_names()
    )

    param_dic = {
        "user" : "onpsmhcjnzdsiz",
        "password" : "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
        "host" : "ec2-3-216-92-193.compute-1.amazonaws.com",
        "port" : "5432", #Postgres Port
        "database" : "dc1fq03u49u20u"
    }

    connect = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
        param_dic['user'],
        param_dic['password'],
        param_dic['host'],
        param_dic['port'],
        param_dic['database']
    )

    engine = create_engine(connect)

    df_vectorizer = pd.DataFrame.from_dict({'used':[tfidf]})

    df_vectorizer.to_sql(
        'vectorizer', 
        con=engine, 
        index=False, 
        if_exists='replace'
    )

    df_ = features.merge(df[target_col], left_index=True, right_index= True)
    
    return df_

def data_to_db(df):
        
    param_dic = {
    "user" : "onpsmhcjnzdsiz",
    "password" : "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
    "host" : "ec2-3-216-92-193.compute-1.amazonaws.com",
    "port" : "5432", #Postgres Port
    "database" : "dc1fq03u49u20u"
    }

    connect = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
        param_dic['user'],
        param_dic['password'],
        param_dic['host'],
        param_dic['port'],
        param_dic['database']
    )
    engine = create_engine(connect)
    df.to_sql(
        'cfh_data', 
        con=engine, 
        index=False, 
        if_exists='replace'
    )
    return df

def from_db(conn_txt_file):
    conn = psycopg2.connect(
        user = "onpsmhcjnzdsiz",
        password = "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
        host = "ec2-3-216-92-193.compute-1.amazonaws.com",
        port = "5432", #Postgres Port
        database = "dc1fq03u49u20u"
    )
    select_query = """
        SELECT *
        FROM cfh_data
    """
    cursor = conn.cursor()
    cursor.execute(select_query)

    tupples = cursor.fetchall()
    cursor.close()
    
    # We just need to turn it into a pandas dataframe
    df = pd.DataFrame(tupples, columns=['category', 'case'])
    return df

    # def data_from_csv(path):

# --- Old Functions

#     df = pd.read_csv(path)
#     return df

# def rescale_numbers(df, scaler = MinMaxScaler):
#     for col in df:
#         if col != 'category':
#             if df[col].dtype in ['int64', 'float64']:
#                 numbers = df[col].astype(float).values.reshape(-1, 1)
#                 df[col] = scaler().fit_transform(numbers)
            
#     return df