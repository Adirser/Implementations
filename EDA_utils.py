#Dependencies

from datetime import datetime
from datetime import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import ppscore as pps
import random
import bokeh
from bokeh.plotting import figure, show
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import tslumen
import tqdm as notebook_tqdm
from tslumen import HtmlReport
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected = True)
import cufflinks as cf
cf.go_offline()
import copy
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
pd.set_option("display.max_columns",80)
pd.set_option("display.max_rows",80)
from statsmodels.tsa.stattools import adfuller,kpss
import torch
from itertools import permutations,combinations,product
import copy



########################################################################
def create_agg(df,min_window,max_window,step,target):
    df = copy.deepcopy(df)
    cols = df.columns
    exclude = []
    for col in cols:
        if (col.startswith("LR")) or (col == target) or (col in ["ts","shift_of_day","day_of_week","month"]):
            print(f"Excluding {col}..")
            exclude.append(col)
    print(exclude)
    for col in df.columns:
        if col not in exclude:
            # print(f"Now handling {col} :")
            for window in range(min_window,max_window,step):
                print(f"Now handling {col} at {window} :")
                df[f"{col}_{window}_mean"] = df[col].rolling(window = window).mean()
                df[f"{col}_{window}_std"] = df[col].rolling(window = window).std()
                # df[f"{col}_{window}_kurtosis"] = df[col].rolling(window = window).kurt()

    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df


def retrieve_lagged_parallel(target:str , min_lag:int, max_lag:int ,step:int, df:pd.DataFrame, forecast_shift:list):
    from itertools import product
    import joblib
    max_corr = {}
    max_pps = {}

    for col in df.columns:
        max_corr[col] = 0
        max_pps[col] = 0
        
    lagged_corr = {}
    lagged_pps = {}

    df = copy.deepcopy(df)
    features  = return_wanted_columns(df,target=target)
    print(f"Total of {len(features)} features to be examined vs the target")

    lags = range(min_lag,max_lag,step)
    print(lags)
    products = product(features,lags,forecast_shift)
 
    out  = joblib.Parallel(n_jobs=-1)(joblib.delayed(preform_calculation)(pd.concat([df[col],df[target]],axis=1),lag,fs,col,target) for col,lag,fs in products)
    for res in out:
        temp_corr,temp_ppz,lag_r,fs,col = res
        if temp_corr > max_corr[col]:
            true_lag = lag_r
            print(true_lag)
            max_corr[col] = copy.deepcopy(temp_corr)
            lagged_corr[col] = {
                "max_correlation":max_corr[col],
                "lag":true_lag,
                "Forecast Shift":fs
            }
        if temp_ppz["ppscore"] > max_pps[col]:
            true_lag = lag_r
            print(f"True LAG for PPS {true_lag}")
            max_pps[col] = temp_ppz["ppscore"]
            lagged_pps[col] = {
                "max_pps":max_pps[col],
                "lag":true_lag,
                "Forecast Shift":fs
            }

    lagged_pps_df = pd.DataFrame(lagged_pps).T
    lagged_corr_df = pd.DataFrame(lagged_corr).T

    lagged_corr = lagged_corr_df.sort_values(by="max_correlation", ascending=False)
    lagged_pps = lagged_pps_df.sort_values(by="max_pps", ascending=False)

    return lagged_corr,lagged_pps

def preform_calculation(df:pd.DataFrame,lag:int,forecast_shift:int,col:str,target:str):
    print(f"Currently {col}(Feature) VS {target}")
    df[target] = df[target].shift(-forecast_shift)
    print(df.head())
    temp_df = pd.DataFrame()
    temp_df[col] = df[col].shift(lag)
    temp_df[target] = df[target]
    temp_df.dropna(inplace=True)
    # Normal Correlation
    corr_score = np.corrcoef(temp_df[col],temp_df[target])[0,1]
    pps_score = pps.score(df ,col,target)
    return corr_score,pps_score,lag,forecast_shift,col


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def create_features(df):
    df = copy.deepcopy(df)
    # Create a new column for the day of the week
    if "ts" not in df.columns:
        df["ts"] = df.index
    df['ts'] = pd.to_datetime(df["ts"])
    df['day_of_week'] = df['ts'].apply(lambda x: datetime.weekday(x))
    # Create a new column for the shift of the day (morning, afternoon, evening)
    df['shift_of_day'] = df['ts'].apply(lambda x: determine_shift_of_day(x))
    df['month'] = pd.DatetimeIndex(df['ts']).month
    return df


def determine_shift_of_day(timestamp):
    print(timestamp)
    time_of_day = datetime.time(timestamp)
    print(time_of_day)
    if time_of_day >= time(6) and time_of_day < time(12):
        return 1
    elif time_of_day >= time(12) and time_of_day < time(18):
        return 2
    elif time_of_day >= time(18) and time_of_day < time(23):
        return 3
    else:
        return 4


def check_feature_stationarity(df):
    results = {}
    for col in df.columns:
        result_adf = adfuller(df[col])
        result_kpss = kpss(df[col])
        
        if result_adf[0] < result_adf[4]["5%"]:
            results[col] = {"P_val_ADF":result_adf[1],"ADF_Stationary":True}
        else:
            results[col] = {"P_val_ADF":result_adf[1],"ADF_Stationary":False}

        if result_kpss[0] > result_kpss[3]["5%"]:
            results[col]["KPSS_Stationary"] = False
            results[col]["P_val_KPSS"] = result_kpss[1]
        else:
            results[col]["KPSS_Stationary"] = True
            results[col]["P_val_KPSS"] = result_kpss[1]
    results = pd.DataFrame(results).T
    return results


def sliding_window(df, window_size, stride,target_col):
    data = df.copy(deep=True)
    train,target = data.drop(columns=[target_col]),data[target_col]
    train,target = train.reset_index(drop=True),target.reset_index(drop=True)
    train = train.to_numpy()
    target = target.to_numpy()
    data_list = []
    target_list = []
    for i in range(0, len(data), stride):
        if i + window_size < len(data):
            data_list.append(train[i:i+window_size])
            target_list.append(target[i+window_size])
        else:
            return torch.tensor(data_list).permute(0,2,1),torch.tensor(target_list)

def bar_plot_feature_importance(df,feature_importance):
    # Create a pandas dataframe with the feature names and importance scores
    df_importance = pd.DataFrame(feature_importance,index = df.columns,columns=["Importance"])
    # Plot the feature importance scores as a count plot
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 8))
    sns.barplot(x=df_importance.index,y="Importance",data=df_importance, palette='Blues_r')
    plt.title('XGBoost Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(rotation = 45)
    plt.show()

def retrieve_changepoints_ts(df,cols_to_check):
    df = copy.deepcopy(df)
    df["ts"] = df.index
    cp_dict = {}
    if cols_to_check is not None:
        for col in cols_to_check:
            cp = np.where((df[col].shift(-1)==df[col])&(df[col].shift(1)!=df[col]))[0]
            cp_dict[col] = cp
            cp_dict["ts"] = df["ts"].iloc[cp]
    else:
        print("Please specify the columns as a list ...")
    return cp_dict
    
def get_window_around_change_point(df, change_point_time, window_size):
    # Calculate the start and end times of the window
    df = copy.deepcopy(df)
    start_time = change_point_time - pd.Timedelta(minutes = window_size)
    end_time = change_point_time + pd.Timedelta(minutes = window_size)
    # Select the rows within the window
    print(f"Start-time : {start_time} \n End-time : {end_time}")
    window_df = df[(df.index >= start_time) & (df.index <= end_time)]
    
    return window_df

def create_operator_activity_plot(df,target,cp):
    fig, ax = plt.subplots()
    plt.axvline(x=cp, color="black", linestyle='--')
    df[target][:cp].plot(x = df.index,y=df[target],ax=ax,color = "red" , label = "Before Change")
    df[target][cp:].plot(x = df.index,y=df[target],ax=ax,color = "green" , label = "After Change")
    ax.set_ylim(df[target].min() - 0.01,df[target].max() + 0.01)
    ax.legend()
    ax.set_title('Operators Activity Before and After')
    ax.set_ylabel(target)
    ax.set_xlabel("Date")
    plt.show()

def tssd_augmentation(df:pd.DataFrame,target:str):
    change_points_ts = retrieve_changepoints_ts(df,[target])
    df = copy.deepcopy(df)
    for ind,t in enumerate(change_points_ts[target][:-1]):
        cp_1 = change_points_ts[target][ind]
        cp_2 = change_points_ts[target][ind+1]
        val_1 = df[target].iloc[cp_1]
        val_2 = df[target].iloc[cp_2]
        slope = (val_2 - val_1)/(cp_2-cp_1)
        li = []
        for ind,t in enumerate(df[target][cp_1:cp_2]):
            li.append(val_1 + slope*ind)
        df[target][cp_1:cp_2] = li
    return df

    # Detecting ourliers under Normal Distribution Assumption
def detect_outliers(df):
    df = copy.deepcopy(df)

    threshold = 3
    mean = np.mean(df)
    std = np.std(df)
    z_scores = [(y - mean) / std for y in df]
    outliers = []
    for i, z_score in enumerate(z_scores):
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

def plot_outliers(df):
    # Extracting Ourliers
    df = copy.deepcopy(df)
    wanted = [x for x in df.columns if x != "ts"]
    outliers_dict = {}

    for col in wanted:
        data = df[col].values
        outliers_dict[col] = detect_outliers(data)

    all_colors=[x for x in plt.cm.colors.cnames.keys()]
    c = random.choices(all_colors, k=len(outliers_dict.keys()))

    # Plot Bars
    counts = [len(outliers_dict[key]) for key in outliers_dict.keys()]
    plt.figure(figsize=(20,8), dpi= 80)
    plt.bar(outliers_dict.keys(), counts, color=c, width=.5)
    for i, val in enumerate(counts):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

    # # Decoration
    plt.gca().set_xticklabels(outliers_dict.keys(), rotation=60, horizontalalignment= 'right')
    plt.title("Outliers counts per feature", fontsize=22)
    plt.ylabel('# of Outliers', fontsize=18)

def fix_outliers(data ,stride = None,window_len = 2500,threshold_up=3,threshold_down=-3):
    # If stride is None then set it to window len
    if stride is None:
        stride = window_len
    # Copy DF
    df = copy.deepcopy(data)
    # Exclude ts / any other unwanted columns
    not_wanted = [x for x in df.columns if x.startswith("LR")]
    not_wanted.append('ts')
    columns_to_change = [col for col in df.columns if col not in not_wanted]
    print(f"About to trim outliers for {columns_to_change} using Windows of {window_len} and Strides of {stride} , Thresholds are set to: {threshold_down,threshold_up}")

    for col in columns_to_change:
        # Clip Negative numbers (to 0) and above 99 Percentile to 0.99 Percentile
        df[col].clip(0.0,df[col].quantile(0.99),inplace = True)
        print(f"Now looking at {col}")
        total_trimmed = []
        for window in range(0,len(df[col]) - stride,stride):
            # Calculate mean and standard deviation of current window
            needs_fixing = df[col].iloc[window:window+window_len]
            mean_ = needs_fixing.mean()
            std_ = needs_fixing.std()
            # Calculate Z-score under Normal Dist Assumption
            z_scores = [(y - mean_) / std_ for y in needs_fixing]
            z_scores = pd.DataFrame(z_scores,index = needs_fixing.index,columns=["z-score"])
            # Indices of outliers
            indices = z_scores[abs(z_scores["z-score"]) >= threshold_up].index
            if len(indices) > 0 :
                df[col][indices] = np.nan
        print(f"Total outliers handled for column {col} : {df[col].isna().sum()}")
    df[columns_to_change] = df[columns_to_change].interpolate(method = "linear")
    df.fillna(method="ffill",inplace=True)
    df.fillna(method="bfill",inplace=True)
    print(f" Total of {df.isna().sum().sum()} NA's (After Interpolation) ")
    return df

# function for random color selection for the plots
def random_color():
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    return color

def split_train_test(df,ratio:float,target_col:str,forecast_shift:int):
    df = copy.deepcopy(df)
    split_point = int(np.floor(len(df)*ratio))
    df[target_col] = df[target_col].shift(-forecast_shift)
    df.dropna(inplace=True)
    X_train = df[:split_point]#.drop(columns = [target_col])
    X_val = df[split_point:]#.drop(columns = [target_col])
    # y_train = df[target_col][:split_point]
    # y_val = df[target_col][split_point:]
    return X_train,X_val


def retrieve_changepoints(df,cols_to_check):
    df = copy.deepcopy(df)
    cp_dict = {}
    if cols_to_check is not None:
        for col in cols_to_check:
            cp = np.where((df[col].shift(-1)==df[col])&(df[col].shift(1)!=df[col]))[0]
            cp_dict[col] = cp
    else:
        print("Please specify the columns as a list ...")
    return cp_dict


# Auto-correlation plot
def plot_auto_correlation(maximum_lag_to_check,df,col):
    df = copy.deepcopy(df)
    #Draw Plot
    fig,ax = plt.subplots(1, 1,figsize=(16,6), dpi= 80)
    plot_pacf(df[col].tolist(), ax=ax, lags=maximum_lag_to_check)
    # lighten the borders
    ax.spines["top"].set_alpha(.3)
    ax.spines["bottom"].set_alpha(.3)
    ax.spines["right"].set_alpha(.3)
    ax.spines["left"].set_alpha(.3)
    # font size of tick labels
    ax.tick_params(axis='both', labelsize=12)
    plt.show()

### Correlation Matrix ###
def plot_correlation_matrix(df):
    df = copy.deepcopy(df)
    plt.figure(figsize=(22,18), dpi= 80)
    sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
    # Decorations
    plt.title('Correlation between variables', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def return_wanted_columns(df,target=None):
    df = copy.deepcopy(df)
    if target is not None:
        wanted = [x for x in df.columns if x not in ["ts",target]]
    else:
        wanted = [x for x in df.columns if x not in ["ts"]]
    return wanted

def plot_scatters(df):
    df = copy.deepcopy(df)
    wanted = return_wanted_columns(df)
    for col in wanted:
        figure = plt.figure(figsize=(20,8))
        plt.scatter(x=df.index,y=df[col],color=random_color(),alpha=0.2,linewidths=1,s=20)
        plt.xlabel("Hourly Data")
        plt.ylabel(f"{col}")
        plt.ylim(bottom = df[col].quantile(0.05),top = df[col].quantile(0.95))

def create_interaction_features(df,degree,target):
    temp_df = copy.deepcopy(df)
    wanted = return_wanted_columns(temp_df,target)
    exclusion = []
    for col_1 in wanted:
        exclusion.append(col_1)
        current = [x for x in wanted if x not in exclusion]
        for deg in range(2,degree+1):
            temp_df[f"{col_1}^{deg}"] = temp_df[col_1].pow(deg)
        for col_2 in current:
            temp_df[f"{col_1}_{col_2}"] = temp_df[col_1]*temp_df[col_2]
    return temp_df.round(5)
            
    
def retrieve_deltas(df,target):
    df = copy.deepcopy(df)
    ts_1 = copy.deepcopy(df[target].shift(1))
    ts_2 = copy.deepcopy(df[target])
    df[f"delta_{target}"] = ts_2 - ts_1
    return df.drop(columns=[target])

def feature_importance(df,threshold,target,estimators):
    df = copy.deepcopy(df)
    selected_features = []
    selected_importances = []
    not_wanted = ["ts",target]
    model = RandomForestRegressor(n_estimators=estimators, random_state=1)
    features = [x for x in df.columns if x not in not_wanted]
    model.fit(df[features],df[target])
    for ind,col in enumerate(features):
        if model.feature_importances_[ind] > threshold:
            selected_features.append(col)
            selected_importances.append(model.feature_importances_[ind])
    ticks = [i for i in range(len(selected_features))]
    all_colors=[x for x in plt.cm.colors.cnames.keys()]
    colors = random.choices(all_colors, k=len(ticks))
    plt.figure(figsize=(20,8))
    plt.bar(ticks, selected_importances,color = colors)
    for i, val in enumerate(np.round(selected_importances,3)):
            plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})
    plt.title("Feature Importance Bar Chart")
    plt.xticks(ticks, selected_features,rotation = 45)
    plt.show()
    print("Selected Features are : ",selected_features)

def retrieve_shifted(df,col,lag):
    df = copy.deepcopy(df)
    df[f"{col}_shifted_{lag}"] = df[col].shift(lag)
    return df
        
def retrieve_most_correlated(lagged_df,threshold,df):
    df = copy.deepcopy(df)
    lagged_dict = {}
    for ind,col in enumerate(lagged_df.index):
        if abs(lagged_df["max_correlation"].iloc[ind]) > threshold:
            lagged_dict[col] =  lagged_df["corr_lag"].iloc[ind]

    for col in lagged_dict.keys():
        df[col] = retrieve_shifted(df,col,lagged_dict[col])
    return df

def fix_dt(df,frequency):
    df = copy.deepcopy(df)
    df.index = df["ts"]
    # df.drop(columns=["ts"],inplace = True)
    df.index = pd.to_datetime(df.index,infer_datetime_format=True)
    df.index = pd.date_range(start=df.index[0], end=df.index[-1], freq=frequency)[:len(df.index)]
    return df

def thresholded_histogram(df,target,num_bins,thresholds,feature):
    df = copy.deepcopy(df)
    print(pd.DataFrame(df[target].describe().iloc[1:]).style.background_gradient("BuPu"))
    feature = feature
    num_bins = num_bins
    thresholds = thresholds
    df["Target"] = pd.cut(df[target] , bins = thresholds,labels=range(len(thresholds) - 1))
    sns.displot(data = df ,x = feature, hue = "Target" , kind = "kde",height= 5 , aspect= 3)


def from_2d_array_to_nested(X, index=None, columns=None, time_index=None, cells_as_numpy=False):
    df = copy.deepcopy(df)
    if (time_index is not None) and cells_as_numpy:
        raise ValueError("`Time_index` cannot be specified when `return_arrays` is True, "
            "time index can only be set to "
            "pandas Series")
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    container = np.array if cells_as_numpy else pd.Series
    # for 2d numpy array, rows represent instances, columns represent time points
    n_instances, n_timepoints = X.shape
    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}
    Xt = pd.DataFrame(pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)]))
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt
        
    
def check_ts_consistency(df,col):
    df = copy.deepcopy(df)
    df["ts"] = pd.to_datetime(df["ts"])
    df['month'] = df['ts'].dt.month
    df['year'] = df['ts'].dt.year
    df['weekday'] = df['ts'].dt.weekday
    df['day'] = df['ts'].dt.day
    df.pivot_table(index='day',columns=['year','month'],aggfunc='count',values='LR00540801BF').style.background_gradient("Greens")





# ######## WORKING OUTLIERS REMOVAL !!! ########################
# def fix_outliers(data ,stride = None,window_len = 2500,threshold_up=3,threshold_down=-3):
#     # If stride is None then set it to window len
#     if stride is None:
#         stride = window_len
#     # Copy DF
#     df = copy.deepcopy(data)
#     # Exclude ts / any other unwanted columns
#     not_wanted = [x for x in df.columns if x.startswith("LR")]
#     not_wanted.append('ts')
#     columns_to_change = [col for col in df.columns if col not in not_wanted]
#     print(f"About to trim outliers for {columns_to_change} using Windows of {window_len} and Strides of {stride} , Thresholds are set to: {threshold_down,threshold_up}")

#     for col in columns_to_change:
#         # Clip Negative numbers (to 0) and above 99 Percentile to 0.99 Percentile
#         df[col].clip(0.0,df[col].quantile(0.99),inplace = True)
#         print(f"Now looking at {col}")
#         total_trimmed = []
#         for window in range(0,len(df[col]) - stride,stride):
#             # Calculate mean and standard deviation of current window
#             needs_fixing = df[col].iloc[window:window+window_len]
#             mean_ = needs_fixing.mean()
#             std_ = needs_fixing.std()
#             # Calculate Z-score under Normal Dist Assumption
#             z_scores = [(y - mean_) / std_ for y in needs_fixing]
#             z_scores = pd.DataFrame(z_scores,index = needs_fixing.index,columns=["z-score"])
#             # Indices of outliers
#             indices = z_scores[abs(z_scores["z-score"]) >= threshold_up].index
#             if len(indices) > 0 :
#                 df[col][indices] = np.nan
#         print(f"Total outliers handled for column {col} : {df[col].isna().sum()}")
#     df[columns_to_change] = df[columns_to_change].interpolate()
#     df.fillna(method="ffill",inplace=True)
#     df.fillna(method="bfill",inplace=True)
#     print(f" Total of {df.isna().sum().sum()} NA's (After Interpolation) ")
#     return df

