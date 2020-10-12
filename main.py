import warnings
warnings.filterwarnings("ignore")

import functools
import streamlit as st

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import plotly.express as px

import scrapper as S
import clientsfh_preprocessing as CP
import regression_models as RM

import glob
import os
import platform
import pickle

import pandas as pd
import numpy as np
import psycopg2

from PIL import Image

# --- Global Variables ---
# global_cat_df = CP.global_cat_df
# global_df_vectorizer = CP.global_df_vectorizer

import nltk
nltk.download('stopwords')

# Streamlit Cache

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def main_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def scrappe_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def upload_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def final_df_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

# def pipe(obj, *fns):
#     return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def hist_of_target_creator(df, target = 'category'):
 
    fig = px.histogram(df, x = target, color= target) #, marginal="rug", hover_data=tips.columns)

    # Plot!
    st.plotly_chart(fig)

    return df

def streamlit_pipe_write_before(df_):
    st.write(f"**Before Over-Under-Sampling**")
    return(df_)
    
def streamlit_pipe_write_after(df_):
    st.write(f"**After Over-Under-Sampling**")
    return(df_)

def streamlit_pipe_write_intro():
    path_web_logo = "web_logo.png"
    web_logo = Image.open(path_web_logo)
    path_cfh_logo = "cfh_logo.png"
    cfh_logo = Image.open(path_cfh_logo)
    

    # st.markdown("<img src='https://raw.githubusercontent.com/sahivy13/clients_from_hell/master/cfh_logo.png' alt='cfh_logo' class='center' />")

    # st.markdown("[![cfh_logo_link](https://raw.githubusercontent.com/sahivy13/clients_from_hell/master/cfh_logo.png)](https://clientsfromhell.net/)")
    st.image(web_logo, width = 85)
    st.markdown("<h1 style='text-align: center; color: black;'>Supervised Machine Learning</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Classifier Algorithm Trainer Bot </h2>", unsafe_allow_html=True)
    st.image(cfh_logo, width = 300)
    st.markdown("<h4 style='text-align: center; color: black;'>Sourced from:  <a class='website-link' href='https://clientsfromhell.net/'>Clients From Hell</a></h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Created by: <a class='website-link' href='https://sahivy.com/'>Sahivy R. Gonzalez</a></h1>", unsafe_allow_html=True)
    
    st.write("")
    

    st.markdown(
        "<p style='text-align: center;'>Each post made on the website is a story from freelancers.</p>",
        unsafe_allow_html=True
        )
    st.markdown(
        "<p style='text-align: center;'>These stories are of bad experiences the freelancers had with clients<br><strong>in real life...</strong> and each story is classified as the following:</p>",
        unsafe_allow_html=True
        )

def graph_instructions():
    st.sidebar.markdown(
        "<h2>Graphs</h2><p style='text-align: left; font-size: small;'>- Click on the categories on the right menu to filter<br>- Hover over columns to see details<br>- Download image graph as png by clicking the camera icon on the image's hover menu</p>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown('***')

def show_df(df):
    st.dataframe(df.style.highlight_max(axis=0))
    return df

def create_db_tables(df_models):
    # Hey future Sahivy, This is past Sahivy.
    # This function requires to add a way to create 
    # the params table based of params used...
    # This is your job, have fun nerd! 
    #      ."".   ."",
    #      |  |  /  /
    #      |  | /  /
    #      |  |/  ;-._
    #      }  ` _/  / ;
    #      |  /` ) /  /
    #      | /  /_/\_/\
    #       (   \ '-  |
    #       \    `.  /
    #        |      |   "SUCKS TO SUCK!"


    conn = psycopg2.connect(
        user = "onpsmhcjnzdsiz",
        password = "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
        host = "ec2-3-216-92-193.compute-1.amazonaws.com",
        port = "5432", #Postgres Port
        database = "dc1fq03u49u20u"
    )

    cur = conn.cursor()

    sql_drop_tables = """
    DROP TABLE IF EXISTS scores_models, params, model_names CASCADE;
    """    

    sql_names_table = """
    CREATE TABLE model_names (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(100) UNIQUE NOT NULL
    );
    """

    sql_params_table = """
    CREATE TABLE params (
        id SERIAL PRIMARY KEY,
        c REAL,
        n_neighbors SMALLINT,
        alpha REAL,
        ccp_alpha REAL,
        max_features VARCHAR(100),
        n_estimators REAL
    );
    """

    sql_main_table = """
    CREATE TABLE scores_models (
     id SERIAL PRIMARY KEY,
     model_name_id SMALLINT NOT NULL,
     best_score SMALLINT NOT NULL,
     best_model BYTEA NOT NULL,
     param_id SMALLINT NOT NULL,
     FOREIGN KEY(param_id) REFERENCES params(id),
     FOREIGN KEY(model_name_id) REFERENCES model_names(id)
    );
    """

    insert_name = """
    INSERT INTO model_names (model_name) 
    VALUES (%s)
    ON CONFLICT (model_name) DO NOTHING;
    """

    commands = [
        sql_drop_tables,
        sql_names_table,
        sql_params_table,
        sql_main_table,
        insert_name
        ]

    list_names = df_models['model_name'].to_list()

    for i, sql in enumerate(commands):
        if i != len(commands)-1:
            cur.execute(sql)
        elif i == 0:
            cur.execute(sql)
        else:
            for name in list_names:
                cur.execute(sql, (name,))


    # close communication with the PostgreSQL database server
    cur.close()
    # commit the changes
    conn.commit()

    return df_models

def pre_saving_crunch(df_models):
    conn = psycopg2.connect(
        user = "onpsmhcjnzdsiz",
        password = "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
        host = "ec2-3-216-92-193.compute-1.amazonaws.com",
        port = "5432", #Postgres Port
        database = "dc1fq03u49u20u"
    )
    names_query = """
        SELECT *
        FROM model_names
    """
    cursor = conn.cursor()
    cursor.execute(names_query)

    name_tuples = cursor.fetchall()
    cursor.close()
    
    # --- ADDING PROPER ID TO EACH MODEL IN THE COPY OF ORIGINAL DF FOR INPUT TO DB---
    df_db_model_names = pd.DataFrame(name_tuples, columns=['id', 'model_name'])
    df_pre_input = df_models.copy()
    df_pre_input['id'] = np.nan
    df_pre_input.loc[df_pre_input.model_name == df_db_model_names.model_name, 'id'] = df_db_model_names.loc[df_db_model_names.model_name == df_pre_input.model_name,'id'].iloc[0]

    # PARAMS TABLE DF
    list_params_name = ['id']
    
    for ix, dict_ in df_models.best_params.iteritems():
        list_params_name.extend(list(dict_.keys()))

    df_params_insert = pd.DataFrame(columns=list_params_name)

    for ix, row in df_pre_input.iterrows():
        dict_input_p = dict()
        dict_input_p['id'] = int(row['id'])
        dict_input_p.update(row['best_params'])
        df_params_insert.append(dict_input_p, ignore_index = True)

    # MAIN TABLE DF
    col_s_m = ['id','model_name_id', 'best_score', 'best_model', 'param_id']
    df_scores_models_insert = pd.DataFrame(columns=col_s_m)

    for ix, row in df_pre_input.iterrows():
        dict_input_p = dict()
        dict_input_p['model_name_id'] = int(row['id'])
        dict_input_p['param_id'] = int(row['id'])
        dict_input_p.update(row[['best_score', 'best_model']].to_dict())
        df_scores_models_insert.append(dict_input_p, ignore_index = True)
    
    return df_pre_input, df_params_insert, df_scores_models_insert

def save_all_or_one(df_params_insert, df_scores_models_insert):

        global_cat_df = pd.read_csv('global_cat_df.csv')
        # global_df_vectorizer = pd.read_csv('global_df_vectorizer.csv')

        conn = psycopg2.connect(
            user = "onpsmhcjnzdsiz",
            password = "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
            host = "ec2-3-216-92-193.compute-1.amazonaws.com",
            port = "5432", #Postgres Port
            database = "dc1fq03u49u20u"
        )

        cur = conn.cursor()
        # creating column list for insertion
        cols_params = ",".join([str(i) for i in df_params_insert.columns.tolist()])
        
        for i,row in df_params_insert.iterrows():
            sql = f"INSERT INTO params ({cols_params}) VALUES ({'%s,'*(len(row)-1)}%s);"
            cur.execute(sql, tuple(row))

        cols_sm = ",".join([str(i) for i in df_scores_models_insert.columns.tolist()])

        for i,row in df_scores_models_insert.iterrows():
            sql = f"INSERT INTO scores_models ({cols_sm}) VALUES ({'%s,'*(len(row)-1)}%s);"
            cur.execute(sql, tuple(row))
        
        cols_cat_sql = ", ".join([str(i) for i in global_cat_df.columns.tolist()])
        cols_cat_table_sql = " SMALLINT PRIMARY KEY, ".join([str(i) for i in global_cat_df.columns.tolist()])+" VARCHAR(100) UNIQUE"

        sql_drop_cat = "DROP TABLE IF EXISTS categories;"
        cur.execute(sql_drop_cat)
        sql_create_cat = f"CREATE TABLE IF NOT EXISTS categories ({cols_cat_table_sql});"
        cur.execute(sql_create_cat)

        for i,row in global_cat_df.iterrows():
            sql = f"INSERT INTO categories ({cols_cat_sql}) VALUES({'%s,'*(len(row)-1)}%s);"
            cur.execute(sql, tuple(row))
        
        with open('tfidf', 'rb') as f:
            load = pickle.load(f)
            f.close()

        tfidf_pickle = pickle.dumps(load)

        df_vectorizer = pd.DataFrame({'tfidf':[tfidf_pickle]}, index = ['pickle_object']).T
        df_vectorizer.index = df_vectorizer.index.set_names(['vec_name'])
        df_vectorizer.reset_index(inplace = True)

        cols_v_sql = ", ".join([str(i) for i in df_vectorizer.columns.tolist()])
        cols_v_table_sql = " VARCHAR(100) PRIMARY KEY, ".join([str(i) for i in df_vectorizer.columns.tolist()])+" BYTEA"

        sql_drop_v = "DROP TABLE IF EXISTS vectorizer;"
        cur.execute(sql_drop_v)
        sql_create_v = f"CREATE TABLE IF NOT EXISTS vectorizer ({cols_v_table_sql});"
        cur.execute(sql_create_v)

        for i,row in df_vectorizer.iterrows():
            sql = f"INSERT INTO vectorizer ({cols_v_sql}) VALUES ({'%s,'*(len(row)-1)}%s);"
            cur.execute(sql, tuple(row))
        

        cur.close()
        # commit the changes
        conn.commit()

# STATING CSS
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# START SIDE BAR
graph_instructions()
st.sidebar.subheader("Pre-Processing:")
use_current_data = st.sidebar.button("Re-Scrape")


# Intro
streamlit_pipe_write_intro()

if use_current_data == False:

    # try:
        
    df_pre_input, df_params_insert, df_scores_models_insert = main_pipe(
        final_df_pipe(
            upload_pipe(
                CP.from_db('db_con.txt'),
                streamlit_pipe_write_before,
                hist_of_target_creator,
                CP.category_replacer,
            ),
            CP.over_under_sampling,
            streamlit_pipe_write_after,hist_of_target_creator,
            CP.convert_to_tfidf,
            RM.best_model,
            show_df
        ),
        create_db_tables,
        pre_saving_crunch
        # df_to_dict,
        # save_all_or_one
    )

    # except:

    #     st.sidebar.write('There was no data in saved in database, thus scrapping website now!')

    #     main_pipe(
    #         final_df_pipe(
    #             scrappe_pipe(
    #                 "https://clientsfromhell.net/",
    #                 S.get_categories,
    #                 S.url_categroy_creator,
    #                 S.page_num_creator,
    #                 S.initialize_scraping,
    #                 CP.df_creator,
    #                 CP.cleaning,
    #                 streamlit_pipe_write_before,
    #                 hist_of_target_creator, 
    #                 CP.data_to_db,
    #                 CP.category_replacer
    #                 ),
    #             CP.over_under_sampling,
    #             streamlit_pipe_write_after,hist_of_target_creator,
    #             CP.convert_to_tfidf,
    #             RM.best_model,
    #             show_df
    #         ),
    #         RM.create_db_tables,
    #         # df_to_dict,
    #         RM.save_all_or_one
    #     )

else:
    
    df_pre_input, df_params_insert, df_scores_models_insert = main_pipe(
        final_df_pipe(
            scrappe_pipe(
                    "https://clientsfromhell.net/",
                    S.get_categories,
                    S.url_categroy_creator,
                    S.page_num_creator,
                    S.initialize_scraping,
                    CP.df_creator,
                    CP.cleaning,
                    streamlit_pipe_write_before,
                    hist_of_target_creator,
                    CP.data_to_db,
                    CP.category_replacer,
            ),
            CP.over_under_sampling,
            streamlit_pipe_write_after,hist_of_target_creator,
            CP.convert_to_tfidf,
            RM.best_model,
            show_df,
        ),
        create_db_tables,
        pre_saving_crunch
        # df_to_dict,
        # save_all_or_one
    )

# # SIDEBAR SELECTION OF MODELS TO SAVE
# model_selection = st.sidebar.multiselect("Choose model to save",tuple(df_pre_input.model_name.values))


# if st.sidebar.button("Save"):
save_all_or_one(df_params_insert, df_scores_models_insert)

# --- SECURITY FUNCTION TO LOOK INTO ---
# if not state.user:
#     username = st.text_input(“User Name:”,value="")
#     password = st.text_input(“Password:”, value="", type=“password”)
#     state.user = authenticate(username, password)
# else:
#     st.write(f"Hey {state.user.name} !")

# --- NOTABLE FUNCTIONS --- 

# if st.sidebar.button("Save Trained Model"):
#         save_all_models('saved_models')

# with st.echo():
#     main_pipe(
#         scrappe_pipe(
#             "https://clientsfromhell.net/",
#             streamlit_pipe_write_paragraph,
#             S.get_categories,
#             S.url_categroy_creator,
#             S.page_num_creator,
#             S.initialize_scraping,
#             CP.df_creator,
#             CP.cleaning,
#             hist_of_target_creator, 
#             CP.category_replacer,
#             CP.data_to_csv
#             ),
#         streamlit_pipe_write_before,
#         hist_of_target_creator,
#         CP.under_sampling_to_2_col_by_index,
#         streamlit_pipe_write_after,hist_of_target_creator,
#         CP.convert_to_tfidf,
#         RM.run_all_models_and_score_k_fold
#     )

# def save_all_models(folder):
#     list_ = ['log_regr.pickle', 'knn.pickle', 'multi.pickle', 'rfc.pickle', 'bernoulli.pickle', 'guassian.pickle']
    
#     for i in list_:
#         os.rename(
#             f"{i}",
#             f"{folder}/{i}"
#         )

# def glob_num_samples_creator():
#     global num_samples
#         num_samples = st.sidebar.slider(
#         "Choose # of samples for each category:",
#         min_value=None,
#         max_value=None,
#         value=mid_num
#     )
#     st.sidebar.markdown("- - -")