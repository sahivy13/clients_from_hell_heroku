import pandas as pd
import numpy as np
# import os
# import copy

import streamlit as st

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

# Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


# Model saving
# import pickle

# Math function needed for models
from math import sqrt

# Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Streamlit
import streamlit as st

# PostgreSQL
# import psycopg2

# # Stating random seed
# np.random.seed(42)

# --- New Added ---

def kfold_cross_validation(df, k = 10, target = 'category'): 
    # --- KFOLD & DATA ---

    kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42)
    features = df[[col for col in df if col != target]]
    target = df[target]   

    # --- CREATING MODELS ---

    def create_models():
        models = list()

        models.append(LogisticRegression(solver = 'lbfgs'))
        models.append(KNeighborsClassifier(n_neighbors = 3)) # k = 5 by default
        models.append(MultinomialNB())
        models.append(RandomForestClassifier(max_depth=10, random_state=42))

        return models

    # --- CROSS_VALIDATION ---

    def cv_eval_model(model, cv, X, y):

        scoring = {
            'R2': make_scorer(r2_score),
            'MSE': make_scorer(mean_squared_error),
            'MAE': make_scorer(mean_absolute_error),
            'Accuracy': make_scorer(accuracy_score),
            'Balanced_Acc': make_scorer(balanced_accuracy_score),
            'Precision': make_scorer(precision_score, average = 'macro'),
            'Recall': make_scorer(recall_score, average = 'macro'),
            'F1': make_scorer(f1_score, average = 'macro'),
        }

        cross_val_obj = cross_validate(model, X, y, scoring = scoring, cv = cv, n_jobs = -1, return_estimator = True)  

        return cross_val_obj # trained_model, 

    # --- RUNNING CROSS-VALIDATION ---

    def run_cross_val(models, cv = kfold, X = features, y = target):
        dict_df = dict()

        for model in models:
            cv_mean = cv_eval_model(model, cv, X, y) #trained_model, 

            # with open (type(model).__name__, 'wb') as f:
            #     pickle.dump(model,f) 

            dict_df[type(model).__name__] = cv_mean

        return dict_df

    models_ = create_models()
    dict_df = run_cross_val(models = models_)
    return dict_df
    
def best_model(df, k = 10, target = 'category'):
    # --- KFOLD & DATA ---

    kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42)
    features = df[[col for col in df if col != target]]
    target = df[target]  

    # --- CREATE MODELS ---

    def create_models():
        models = list()

        models.append(LogisticRegression(solver = 'lbfgs'))
        models.append(KNeighborsClassifier()) # k = 5 by default
        models.append(MultinomialNB())
        models.append(RandomForestClassifier(random_state=42))

        return models

    # --- SCORES ---

    scoring = {
    'R2': make_scorer(r2_score),
    'MSE': make_scorer(mean_squared_error),
    'MAE': make_scorer(mean_absolute_error),
    'Accuracy': make_scorer(accuracy_score),
    'Balanced_Acc': make_scorer(balanced_accuracy_score),
    'Precision': make_scorer(precision_score, average = 'macro'),
    'Recall': make_scorer(recall_score, average = 'macro'),
    'F1': make_scorer(f1_score, average = 'macro'),
    }

    # --- PARAMETERS ---
        # LR
    Cs = [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100]

        # KNN
    n_neighbors_ = [3, 5, 8]
        # MultiNB
    alphas = [0, 0.5, 1.0]
        # RandomForest
    n_estimators_ = [50, 100, 150]
    max_features_ = ['sqrt', 'log2']
    ccp_alphas = [0, 0.5, 1]

    dict_param_grid = {
        'LogisticRegression': {'C': Cs},
        'KNeighborsClassifier': {'n_neighbors': n_neighbors_},
        'MultinomialNB': {'alpha': alphas},
        'RandomForestClassifier': {'n_estimators': n_estimators_, 'max_features': max_features_, 'ccp_alpha': ccp_alphas}
    }

    # --- RUNNING GRIDSEARCHCV
    models = create_models()

    df_models = pd.DataFrame(columns = ['model_name', 'best_params', 'best_score', 'best_model'])

    # --- PROGRESS BAR ---
    my_bar = st.progress(0)

    for i, model in enumerate(models):
        param_grid = dict_param_grid[type(model).__name__]
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv = kfold,
            scoring = scoring,
            refit = 'Accuracy',
            n_jobs = -1
        )
        grid_search.fit(features, target)

        df_input = pd.DataFrame({'model_name':[f'{type(model).__name__}'], 'best_params':[grid_search.best_params_], 'best_score':[grid_search.best_score_], 'best_model': [grid_search.best_estimator_]})
        df_models = pd.concat([df_models,df_input], axis = 0, sort = False).reset_index(drop = True)

        my_bar.progress((i+1)/len(models))

    return df_models

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
# def create_db_tables(df_models):
#     # Hey future Sahivy, This is past Sahivy.
#     # This function requires to add a way to create 
#     # the params table based of params used...
#     # This is your job, have fun nerd! 
#     #      ."".   ."",
#     #      |  |  /  /
#     #      |  | /  /
#     #      |  |/  ;-._
#     #      }  ` _/  / ;
#     #      |  /` ) /  /
#     #      | /  /_/\_/\
#     #       (   \ '-  |
#     #       \    `.  /
#     #        |      |   "SUCKS TO SUCK!"


#     conn = psycopg2.connect(
#         user = "onpsmhcjnzdsiz",
#         password = "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
#         host = "ec2-3-216-92-193.compute-1.amazonaws.com",
#         port = "5432", #Postgres Port
#         database = "dc1fq03u49u20u"
#     )

#     cur = conn.cursor()

#     sql_drop_tables = """
#     DROP TABLE IF EXISTS scores_models, params, model_names CASCADE;
#     """    

#     sql_names_table = """
#     CREATE TABLE model_names (
#         id SERIAL PRIMARY KEY,
#         model_name VARCHAR(100) UNIQUE NOT NULL
#     );
#     """

#     sql_params_table = """
#     CREATE TABLE params (
#         id SERIAL PRIMARY KEY,
#         c REAL,
#         n_neighbors SMALLINT,
#         alpha REAL,
#         ccp_alpha REAL,
#         max_features VARCHAR(100),
#         n_estimators REAL
#     );
#     """

#     sql_main_table = """
#     CREATE TABLE scores_models (
#      id SERIAL PRIMARY KEY,
#      model_name_id SMALLINT NOT NULL,
#      best_score SMALLINT NOT NULL,
#      best_model BYTEA NOT NULL,
#      param_id SMALLINT NOT NULL,
#      FOREIGN KEY(param_id) REFERENCES params(id),
#      FOREIGN KEY(model_name_id) REFERENCES model_names(id)
#     );
#     """

#     insert_name = """
#     INSERT INTO model_names (model_name) 
#     VALUES (%s)
#     ON CONFLICT (model_name) DO NOTHING;
#     """

#     commands = [
#         sql_drop_tables,
#         sql_names_table,
#         sql_params_table,
#         sql_main_table,
#         insert_name
#         ]

#     list_names = df_models['model_name'].to_list()

#     for i, sql in enumerate(commands):
#         if i != len(commands)-1:
#             cur.execute(sql)
#         elif i == 0:
#             cur.execute(sql)
#         else:
#             for name in list_names:
#                 cur.execute(sql, (name,))


#     # close communication with the PostgreSQL database server
#     cur.close()
#     # commit the changes
#     conn.commit()

#     return df_models

# def save_all_or_one(df_models):

#     conn = psycopg2.connect(
#         user = "onpsmhcjnzdsiz",
#         password = "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
#         host = "ec2-3-216-92-193.compute-1.amazonaws.com",
#         port = "5432", #Postgres Port
#         database = "dc1fq03u49u20u"
#     )
#     names_query = """
#         SELECT *
#         FROM model_names
#     """
#     cursor = conn.cursor()
#     cursor.execute(names_query)

#     name_tuples = cursor.fetchall()
#     cursor.close()
    
#     # ADDING PROPER ID TO EACH MODEL
#     df_db_model_names = pd.DataFrame(name_tuples, columns=['id', 'model_name'])
#     df_pre_input = df_models.copy()
#     df_pre_input['id'] = np.nan
#     df_pre_input.loc[df_pre_input.model_name == df_db_model_names.model_name, 'id'] = df_db_model_names.loc[df_db_model_names.model_name == df_pre_input.model_name,'id'].iloc[0]

#     model_selection = st.sidebar.multiselect("Choose model to save",tuple(df_pre_input.model_name.values))

#     list_params_name = ['id']
    
#     for ix, dict_ in df_models.items('best_params'):
#         list_params_name.extend(list(dict_.keys()))

#     df_params_insert = pd.DataFrame(columns=list_params_name)

#     for ix, row in df_pre_input.iterrows():
#         dict_input_p = dict()
#         dict_input_p['id'] = row['id']
#         dict_input_p.update(row['best_params'])
#         df_params_insert.append(dict_input_p, ignore_index = True)

#     col_s_m = ['id','model_name_id', 'best_score', 'best_model', 'param_id']
#     df_scores_models_insert = pd.DataFrame(columns=col_s_m)

#     for ix, row in df_pre_input.iterrows():
#         dict_input_p = dict()
#         dict_input_p['model_name_id'] = row['id']
#         dict_input_p['param_id'] = row['id']
#         dict_input_p.update(row[['best_score', 'best_model']].to_dict('r')[0])
#         df_scores_models_insert.append(dict_input_p, ignore_index = True)
    

#     if st.sidebar.button("Save"):

#         conn = psycopg2.connect(
#             user = "onpsmhcjnzdsiz",
#             password = "54cad954a541572fa5b79d1cd9448b4c2971306246824c1f9468c853ef6471b0",
#             host = "ec2-3-216-92-193.compute-1.amazonaws.com",
#             port = "5432", #Postgres Port
#             database = "dc1fq03u49u20u"
#         )

#         cur = conn.cursor()
#         # creating column list for insertion
#         cols_params = ",".join([str(i) for i in df_params_insert.columns.tolist()])
        
#         for i,row in df_params_insert.iterrows():
#             sql = f"INSERT INTO params ({cols_params}) VALUES ({'%s,'*(len(row)-1)}%s);"
#             cur.execute(sql, tuple(row))

#         cols_sm = ",".join([str(i) for i in df_scores_models_insert.columns.tolist()])

#         for i,row in df_scores_models_insert.iterrows():
#             sql = f"INSERT INTO scores_models ({cols_sm}) VALUES ({'%s,'*(len(row)-1)}%s);"
#             cur.execute(sql, tuple(row))

#         cur.close()
#         # commit the changes
#         conn.commit()

# --- Old Functions ---

# def all_bool_models_fitting(X_train, y_train):

#     if os.path.isfile('guassian.pickle'):

#         list_ = ['bernoulli.pickle', 'guassian.pickle']

#         with open (list_[0], 'rb') as f :
#             bernoulli = pickle.load(f)

#         with open (list_[1], 'rb') as f :
#             guassian = pickle.load(f)

#         bernoulli.fit(X_train, y_train.values.ravel())
#         guassian.fit(X_train, y_train.values.ravel())

#     else: 

#         bernoulli = BernoulliNB().fit(X_train, y_train.values.ravel())
#         # d_bernoulli = copy.deepcopy(bernoulli)
        
#         guassian = GaussianNB().fit(X_train, y_train.values.ravel())
#         # d_guassian = copy.deepcopy(guassian)
        
#         list_ = [('bernoulli.pickle', bernoulli), ('guassian.pickle', guassian)]

#         for mod in list_:
#             with open (mod[0], 'wb') as f:
#                 pickle.dump(mod[1],f)
            
#             # with open (mod[0], 'rb') as f :
#             #     mod[1] = pickle.load(f)
    
#     return bernoulli, guassian

# def all_num_models_fitting(X_train, y_train):

    # if os.path.isfile('rfc'):

    #     list_ = ['log_regr.pickle', 'knn.pickle', 'multi.pickle', 'rfc.pickle']

    #     with open (list_[0], 'rb') as f :
    #         log_regr = pickle.load(f)

    #     with open (list_[1], 'rb') as f :
    #         knn = pickle.load(f)

    #     with open (list_[2], 'rb') as f :
    #         multi = pickle.load(f)

    #     with open (list_[3], 'rb') as f :
    #         rfc = pickle.load(f)

    #     log_regr.fit(X_train, y_train.values.ravel())
    #     knn.fit(X_train, y_train.values.ravel())
    #     multi.fit(X_train, y_train.values.ravel())
    #     rfc.fit(X_train, y_train.values.ravel())

    # else:
    #     log_regr = LogisticRegression(solver = 'lbfgs')
    #     log_regr.fit(X_train, y_train.values.ravel())
    #     # d_log_regr = copy.deepcopy(log_regr)

    #     knn = KNeighborsClassifier(n_neighbors = 3) # k = 5 by default
    #     knn.fit(X_train, y_train.values.ravel())
    #     # d_knn = copy.deepcopy(knn)

    #     multi = MultinomialNB()
    #     multi.fit(X_train, y_train.values.ravel())
    #     # d_multi = copy.deepcopy(multi)
        
    #     rfc = RandomForestClassifier(max_depth=10, random_state=42)
    #     rfc.fit(X_train, y_train.values.ravel())
    #     # d_rfc = copy.deepcopy(rfc)

    #     list_ = [('log_regr.pickle', log_regr), ('knn.pickle', knn), ('multi.pickle', multi), ('rfc.pickle', rfc)]

    #     for mod in list_:
    #         with open (mod[0], 'wb') as f:
    #             pickle.dump(mod[1],f) 

    #         # with open (mod[0], 'rb') as f :
    #         #     mod[1] = pickle.load(f)
    
    # return log_regr, knn, multi, rfc

# def evaluate_model(model, train_X, test_X, train_y, test_y):
    
#     model = model.fit(train_X, train_y)
#     score = model.score(test_X, test_y)
#     #     print(f"Accuracy: {round(score, 2)}")
#     return model, score

# def predict(model, X_test):
#     prediction = model.predict(X_test)
#     return prediction

# def k_fold_score_new(df, model_name, target = 'category'):
#     scores = []
#     r2_scores = []
#     mse_scores = []
#     rmse_scores = []
#     mae_scores = []
#     acc_scores = []
#     bacc_scores = []
#     prec_scores = []
#     rec_scores = []
#     f1_scores = []

#     features = df[[col for col in df if col != target]]
#     target = df[target]
    
#     n= 10
#     kf = KFold(n_splits = n, random_state = 42)
#     for train_i, test_i in kf.split(df):
        
#         X_train = features.iloc[train_i]
#         X_test = features.iloc[test_i]
#         y_train = target.iloc[train_i]
#         y_test = target.iloc[test_i]
        
#         X_train_bool = X_train.astype('bool')
#         X_test_bool = X_test.astype('bool')
        
#         models = all_num_models_fitting(X_train, y_train) #log_regr, knn, multi, rfc
#         models = models + all_bool_models_fitting(X_train_bool, y_train) #adding ber, gau
        


#         model_names = {
#             '**Log Regression**': 0, '**KNN**': 1,
#             '**Multinomial**': 2, '**Random Forest**': 3,
#             '**Bernoulli**': 4, '**Gaussian**': 5
#         }

#         # model_names.get(model_name)

#         if model_names.get(model_name) < 4:

#             model = models[model_names.get(model_name)]
#             score = model.score(X_test, y_test) #returns score 
#             scores.append(score)

#             prediction = predict(models[model_names.get(model_name)], X_test)

#             r2 = r2_score(y_test, prediction)
#             r2_scores.append(r2)

#             mse = mean_squared_error(y_test, prediction)
#             mse_scores.append(mse)

#             rmse = sqrt(mse)
#             rmse_scores.append(rmse)

#             mae = mean_absolute_error(y_test, prediction)
#             mae_scores.append(mae)

#             acc = accuracy_score(y_test, prediction)
#             acc_scores.append(acc)

#             bacc = balanced_accuracy_score(y_test, prediction)
#             bacc_scores.append(bacc)

#             prec = precision_score(
#                 y_test,
#                 prediction,
#                 pos_label = 10,
#                 average = 'weighted'
#             )
#             prec_scores.append(prec)
            
#             rec = recall_score(
#             y_test,
#             prediction,
#             pos_label = 10,
#             average = 'weighted'
#             )
#             rec_scores.append(rec)

#             f1 = f1_score(
#                 y_test,
#                 prediction,
#                 pos_label = 10,
#                 average = 'weighted'
#             )
#             f1_scores.append(f1)
            
#             # return r2, mse, rmse, mae, acc, bacc, prec, rec, f1
#         else:
#             score = models[model_names.get(model_name)].score(X_test_bool, y_test) #returns score 
#             scores.append(score)

#             prediction = predict(models[model_names.get(model_name)], X_test_bool)

#             r2 = r2_score(y_test, prediction)
#             r2_scores.append(r2)

#             mse = mean_squared_error(y_test, prediction)
#             mse_scores.append(mse)

#             rmse = sqrt(mse)
#             rmse_scores.append(rmse)

#             mae = mean_absolute_error(y_test, prediction)
#             mae_scores.append(mae)

#             acc = accuracy_score(y_test, prediction)
#             acc_scores.append(acc)

#             bacc = balanced_accuracy_score(y_test, prediction)
#             bacc_scores.append(bacc)

#             prec = precision_score(
#                 y_test,
#                 prediction,
#                 pos_label = 2,
#                 average = 'weighted'
#             )
#             prec_scores.append(prec)
            
#             rec = recall_score(
#             y_test,
#             prediction,
#             pos_label = 2,
#             average = 'weighted'
#             )
#             rec_scores.append(rec)

#             f1 = f1_score(
#                 y_test,
#                 prediction,
#                 pos_label = 2,
#                 average = 'weighted'
#             )
#             f1_scores.append(f1)

#     def avg_score(x):
#         avg = sum(x)/len(x)
#         return avg

#     score_avg = avg_score(scores)
#     r2_avg = avg_score(r2_scores)
#     mse_avg= avg_score(mse_scores)
#     rmse_avg = avg_score(rmse_scores)
#     mae_avg = avg_score(mae_scores)
#     acc_avg = avg_score(acc_scores)
#     bacc_avg = avg_score(bacc_scores)
#     prec_avg = avg_score(prec_scores)
#     rec_avg = avg_score(rec_scores)
#     f1_avg = avg_score(f1_scores)

#     return (
#         score_avg, r2_avg, mse_avg,
#         rmse_avg, mae_avg, acc_avg,
#         bacc_avg, prec_avg, rec_avg, f1_avg
#         )

# def run_all_models_and_score_k_fold(df):
    
#     model_names = ['**Log Regression**', '**KNN**', '**Multinomial**', '**Random Forest**', '**Bernoulli**', '**Gaussian**']
    
#     metrics_names = [
#         'R2: ', 'MSE: ', 'RMSE: ', 'MAE: ',
#         'Accuracy: ', 'Balanced Acc: ', 'Precision: ',
#         'Recall: ', 'F1 Score: '
#     ]

#     for i, model in enumerate(model_names):
#         if i < 4:

#             st.write(model_names[i])
#             st.write('')
#             st.write('kfold score: ',k_fold_score_new(df, model_names[i])[0]*100,'%') #prints score kfold
#             st.write('')

#             for j, name in enumerate(metrics_names):
#                   st.write(name, k_fold_score_new(df, model_names[i])[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1

#             st.write('')

#         else:

#             st.write(model_names[i])
#             st.write('')
#             st.write('kfold score: ',k_fold_score_new(df, model_names[i])[0]*100,'%') #prints score kfold
#             st.write('')
            
#             for j, name in enumerate(metrics_names):
#                   st.write(name, k_fold_score_new(df, model_names[i])[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1
#             st.write('')

# def new_run_all_models_and_score_k_fold(df):
    
#     model_names = ['**Log Regression**', '**KNN**', '**Multinomial**', '**Random Forest**', '**Bernoulli**', '**Gaussian**']
    
#     metrics_names = [
#         'R2: ', 'MSE: ', 'RMSE: ', 'MAE: ',
#         'Accuracy: ', 'Balanced Acc: ', 'Precision: ',
#         'Recall: ', 'F1 Score: '
#     ]

#     # df_metrics = pd.DataFrame(
#     #     columns= [
#     #         'Model', 'KFold_Score', 'R2: ',
#     #         'MSE', 'RMSE', 'MAE','Accuracy',
#     #         'Balanced_Acc', 'Precision:', 'Recall', 'F1_Score'
#     #     ]
#     # )

#     for i, model_n in enumerate(model_names):

#         # model_metrics_dict = {}

#         metrics_i = k_fold_score_new(df, model_names[i])

#         # if i < 4:

#         st.write(model_n)
#         st.write('')
#         st.write('kfold score: ',metrics_i[0]*100,'%') #prints score kfold
#         st.write('')

#         for j, name in enumerate(metrics_names):
#             st.write(name, metrics_i[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1

#             st.write('')

#         # else:

#         #     st.write(model_n)
#         #     st.write('')
#         #     st.write('kfold score: ',metrics_i[0]*100,'%') #prints score kfold
#         #     st.write('')
            
#         #     for j, name in enumerate(metrics_names):
#         #           st.write(name, metrics_i[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1
#         #     st.write('')