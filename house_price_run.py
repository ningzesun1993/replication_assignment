import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import time
import random
import json
random.seed(0)



def make_int(x):
    try:
        return int(x)
    except:
        return np.nan
    
    
def floor_type(x):
    try:
        return x.split(' ')[0]
    except:
        return np.nan


def floor_height(x):
    try:
        return make_int(x.split(' ')[1])
    except:
        return np.nan
    
    
    
def outlier_removal(df, i):
    q = df[i].quantile([0.25,0.75])
    q1, q3 = q[0.25], q[0.75]
    iqr = q3 - q1
    return [i, q1 - 1.5 * iqr, q3 + 1.5 * iqr]



def drop_columns(df, drop):
    return df[[x for x in df.columns if x not in drop]]



def run_regression(m, t_list, name, res_list, verbose = True):
    '''
    res_list = [train, test, rmse_train, rmse_test, name]
    '''
    now = time.time()
    train_x, test_x, train_y, test_y = t_list
    m.fit(train_x, train_y)
    test = m.predict(test_x)
    train = m.predict(train_x)
    rmse = mean_squared_error(test, test_y, squared = False)
    rmse_t = mean_squared_error(train, train_y, squared = False)
    res_list[0].append(train)
    res_list[1].append(test)
    res_list[2].append(rmse)
    res_list[3].append(rmse_t)
    res_list[4].append(name)
    p_res = f"For {name}, time is {round(time.time() - now,4)}, test RMSLE"
    p_res += f" is {round(rmse,4)}, train RMSLE is {round(rmse_t,4)}"
    if verbose:
        print(p_res)
    return res_list, m


def regen_model(rf_params, lgb_params, xgb_params):
    rf = RandomForestRegressor(**rf_params)
    lgb_m = lgb.LGBMRegressor(**lgb_params)
    xgb_m = xgb.XGBRegressor(**xgb_params)
    return rf, lgb_m, xgb_m



def final_dataframe(df_train, columns):
    df_train = np.concatenate([np.expm1(np.expand_dims(x, axis = 1)) for x in df_train], axis = 1)
    df_train = pd.DataFrame(df_train, columns = columns)
    return df_train



def grid_fit(clf, params_range, t_list):
    grid_clf = GridSearchCV(clf, params_range)
    grid_clf.fit(t_list[0], t_list[2])
    return grid_clf.best_estimator_


class np_encode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
        
        
        

if __name__ == '__main__':
    df = pd.read_csv('./resources/house_price.csv',encoding='gbk',low_memory=False)
    df = drop_columns(df, ['DOM', 'url', 'id', 'Cid'])
    int_list = ['livingRoom', 'bathRoom', 'drawingRoom', 'constructionTime']
    for i in int_list:
        df[i] = df[i].apply(make_int)
    df['tradeTime'] = pd.to_datetime(df.tradeTime)
    df['tradeTime'] = df.tradeTime.dt.year
    df['floorHeight'] = df.floor.apply(floor_height)
    df['floorType'] = LabelEncoder().fit_transform(df.floor.apply(floor_type))
    df['age'] = 2019 - df.constructionTime
    df = df.loc[df.price >= 14000,:]
    df = df.dropna()
    location = df[['Lat', 'Lng']].to_numpy()
    point = np.ones_like(location) * np.array([39.916668, 116.383331])
    df['distance'] = np.linalg.norm(location - point, axis = 1)
    num_factor = ["followers", "square", "age", "floorHeight", "distance", 
                   "ladderRatio", "communityAverage", "totalPrice"]
    outlier = []
    for i in num_factor:
        outlier.append(outlier_removal(df, i))
    for i in outlier:
        df = df.loc[(df[i[0]] <= i[2]) & (df[i[0]] >= i[1]),:]
    drop_column = ["floor", "constructionTime", "totalPrice", "kitchen", "bathRoom", "drawingRoom", "totalPrice"]
    df = drop_columns(df, drop_column).reset_index(drop = True)
    df['price_t'] = np.log1p(df.price)
    dummy_columns = ["district", "buildingType", "renovationCondition", "elevator", "subway",
                 "fiveYearsProperty" , "floorType", "buildingStructure"]
    df_final = drop_columns(df, dummy_columns).reset_index(drop = True)
    for i in dummy_columns:
        df[i] = LabelEncoder().fit_transform(df[i])
        df_t = pd.get_dummies(df[i])
        df_t = df_t.rename(columns = {x: f"{i}_{x}" for x in range(len(df_t.columns))})
        df_final = pd.concat([df_final, df_t], axis = 1)
    t_list = train_test_split(drop_columns(df_final, ['price', 'price_t']), 
                           df_final.price_t, test_size= 0.1, random_state = 0)
    params = {}
    rf_params = {'max_depth': 20, 'min_samples_split': 10, 'n_jobs': -1}
    rf_init = {'n_jobs': -1}
    rf_range = {'max_depth': [15, 25], 'min_samples_split': [7, 13]}
    lgb_params = {'objective': 'regression', 'learning_rate': 0.15, 'n_estimators': 64, 
                  'min_child_weight': 2, 'num_leaves': 36, 'colsample_bytree': 0.8, 
                  'reg_lambda': 0.40}
    lgb_init = {'objective': 'regression', 'reg_lambda': 0.40, 'colsample_bytree': 0.8, 
                'min_child_weight': 2, 'colsample_bytree': 0.8}
    lgb_range = {'learning_rate': [0.1, 0.2], 'n_estimators': [32,128], 'num_leaves': [16, 64]}
    xgb_params = {'objective': 'reg:squarederror', 'min_child_weight': 2, 'subsample': 1,
                  'colsample_bytree': 0.8, 'learning_rate': 0.1, 'n_estimators': 500, 'nthread': 16,
                  'reg_lambda': 0.45, 'reg_alpha': 0, 'gamma': 0.5, 'tree_method': 'gpu_hist'}
    xgb_init = {'objective': 'reg:squarederror', 'reg_lambda': 0.45, 'min_child_weight': 2, 
                'gamma': 0.5, 'tree_method': 'gpu_hist', 'colsample_bytree': 0.8, 'nthread': 16}
    xgb_range = {'learning_rate': [0.05, 0.2], 'n_estimators': [400, 600]}
    res = [[t_list[2]], [t_list[3]], [0], [0], ['target']]
    rf_m, lgb_m, xgb_m = regen_model(rf_params, lgb_params, xgb_params)
    res, rf_m = run_regression(rf_m, t_list, 'Random_Forest', res)
    res, lgb_m = run_regression(lgb_m, t_list, 'lightGBM', res)
    res, xgb_m = run_regression(xgb_m, t_list, 'xgboost', res)
    rf_m, lgb_m, xgb_m = regen_model(rf_init, lgb_init, xgb_init)
    rf_m = grid_fit(rf_m, rf_range, t_list)
    res, rf_m = run_regression(rf_m, t_list, 'Random_Forest_grid_search', res)
    params['rf_grid'] = rf_m.get_params()
    lgb_m = grid_fit(lgb_m, lgb_range, t_list)
    res, lgb_m = run_regression(lgb_m, t_list, 'LightGBM_grid_search', res)
    params['lgb_grid'] = lgb_m.get_params()
    xgb_m = grid_fit(xgb_m, xgb_range, t_list)
    res, xgb_m = run_regression(xgb_m, t_list, 'XGboost_grid_search', res)
    params['xgb_grid'] = xgb_m.get_params()
    vr_range = {'rf__max_depth': [18, 22], 'lgb__n_estimators': [32, 40], 
                'lgb__num_leaves': [30, 40], 'xgb__n_estimators': [480, 520]}
    lgb_init['learning_rate'] = 0.15
    xgb_init['learning_rate'] = 0.1
    rf_init['min_samples_split'] = 10
    rf, lgb_m, xgb_m = regen_model(rf_params, lgb_params, xgb_params)
    hybrid_m = VotingRegressor([('rf', rf), ('lgb', lgb_m), ('xgb', xgb_m)])
    res, hybrid_m = run_regression(hybrid_m, t_list, 'hybrid_regrission', res)
    rf, lgb_m, xgb_m = regen_model(rf_init, lgb_init, xgb_init)
    hybrid_m = VotingRegressor([('rf', rf), ('lgb', lgb_m), ('xgb', xgb_m)])
    hybrid_m = grid_fit(hybrid_m, vr_range, t_list)
    res, hybrid_m = run_regression(hybrid_m, t_list, 'hybrid_regrission_grid_search', res)
    params['vr_grid'] = {x[0]: x[1].get_params() for x in hybrid_m.get_params()['estimators']}
    rf, lgb_m, xgb_m = regen_model(rf_params, lgb_params, xgb_params)
    stack_m = StackingRegressor(estimators = [('rf', rf), ('lgb', lgb_m)], final_estimator = xgb_m)#('xgb', xgb_m))
    res, stack_m = run_regression(stack_m, t_list, 'stack_generation', res)
    rf, lgb_m, xgb_m = regen_model(rf_init, lgb_init, xgb_params)
    stack_m = StackingRegressor(estimators = [('rf', rf), ('lgb', lgb_m)], final_estimator = xgb_m)
    sr_range = {x: vr_range[x] for x in vr_range.keys() if 'xgb' not in x}
    stack_m = grid_fit(stack_m, sr_range, t_list)
    params['sr_grid'] = {x[0]: x[1].get_params() for x in stack_m.get_params()['estimators']}
    res, stack_m = run_regression(stack_m, t_list, 'stack_generation_grid_search', res)
    df_train = final_dataframe(res[0], res[-1])
    df_test = final_dataframe(res[1], res[-1])
    df_train.to_csv('./resources/train_result.csv', index = False)
    df_test.to_csv('./resources/test_result.csv', index = False)
    df_train_r = pd.DataFrame([res[4], res[2], res[3]]).T.iloc[1:]
    df_train_r.columns = ['models', 'test_rmsle', 'train_rmsle']
    df_train_r.to_csv('./resources/rmse_result.csv', index = False)
    with open('/resourcesparams.json', 'w') as wfile:
        json.dump(params, wfile, cls = np_encode)