import pandas as pd
import datetime
from datetime import timedelta
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from pprint import pprint
yf.pdr_override()

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn import preprocessing, cross_validation, svm

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)

import numpy as np
from sklearn.model_selection import train_test_split

import time
start_time = time.time()
print("program start time : ", start_time)

def get_eod_data_yf(stock_name, start_date, end_date, save_to_csv=""):
    """The Function gets Dynamic end of day stock market data from yahoo finance api"""
    try:
        # download stock data and place in DataFrame
        df = pdr.get_data_yahoo(stock_name, start=start_date, end=end_date)
        # Save the output to csv file if required
        if save_to_csv:
            df.to_csv(save_to_csv)
        return df
    except Exception as e:
        print("Excpetion in downloading data : ",e)
        pass

def get_eod_data_csv(filename):
    """ This Function reads End of Day Data from a csv file specified """
    df = pd.read_csv(filename)
    print("records fetched : ",len(df))
    print(df.head())
    return df

def get_tickbytick_data_mongodb(db_connection, ticker="ACC"):
    """ This Function reads Tick by Tick Data from """
    print("***** Getting Tick By Tick Data from MongoDB *****")

def get_eod_data_mongodb(db_connection, ticker="ACC"):
    """ This Function reads end of day data from mongodb """
    print("***** Getting End of Day Data from MongoDB *****",ticker)
    try:
        out=pd.DataFrame(list(db_connection.main_data_testing.find({"symbol":ticker})))
        print("***** EOD from MongoDB ******")
        print(out.head())
        return out
    except Exception as e:
        print("Exception in getting data from MongoDB : ",e)
        return pd.DataFrame()

def computeClassification(actual):
    if (actual > 0):
        return 1
    else:
        return -1

# RSI
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    rs = pd.stats.moments.ewma(u, com=period - 1, adjust=False) / \
         pd.stats.moments.ewma(d, com=period - 1, adjust=False)
    return 100 - 100 / (1 + rs)

def apply_logistic_regression(train_x, train_y, test_x, test_y):
    """ This Functions apply logistic regression on the following Training Dataset """
    print("******* Predicting Values Using Logistic Regression ********")

    # Importing Logistic Regression
    from sklearn.linear_model import LogisticRegression

    # Intiating Logistic Regression
    clf = LogisticRegression()

    # Fitting the Model with Training Data
    clf.fit(train_x, train_y)

    # Predicting test data
    predictions = clf.predict(test_x)  # predictions is an array containing the predicted values (-1 or 1) for the features in data_X_test

    # Finding Accuracy Score
    # accuracy_scor = accuracy_score(test_y,predictions)
    # r2_scor = r2_score(test_y,predictions)

    # print("accuracy_scor : ",accuracy_scor)
    # print("r2_scor       : ",r2_scor)

    return predictions

def apply_linear_regression(train_x, train_y, test_x, test_y):
    """ This Functions apply linear regression on the following Training Dataset """
    print("******* Predicting Values Using Linear Regression ********")

    # Importing Logistic Regression
    from sklearn.linear_model import LinearRegression

    # Intiating Logistic Regression
    clf = LinearRegression()

    # Fitting the Model with Training Data
    clf.fit(train_x, train_y)

    # Predicting test data
    predictions = clf.predict(test_x)  # predictions is an array containing the predicted values (-1 or 1) for the features in data_X_test

    # Finding Accuracy Score
    # accuracy_scor = accuracy_score(test_y,predictions)
    # r2_scor = r2_score(test_y,predictions)

    # print("accuracy_scor : ",accuracy_scor)
    # print("r2_scor       : ",r2_scor)
    return predictions

def apply_SVR(train_x, train_y, test_x):
    """This Functions apply Support Vector Classifier on the following training dataset """

    # Importing SVR Classifier
    from sklearn.svm import LinearSVR

    # Initiating SVC Classifier
    clf = LinearSVR()

    # Fitting into Model
    clf.fit(train_x, train_y)
    print("******* Predicting Values Using SVC ********")

    # Predicting the values for test data
    predictions = clf.predict(test_x)

    # accu_score = accuracy_score(test_y, predictions)
    # print("accu_score : ",accu_score)

    # r2_scor = r2_score(test_y, predictions)
    # print("r2_scor    : ",r2_scor)

    return predictions

def apply_gradient_boost(train_x,train_y,test_x):
    """This Functions apply Gradient Boost on the following training dataset"""

    print("******* Predicting Values Using Gradient Boost ********")
    # Importing Gradient Boost Classifier
    from sklearn.ensemble import GradientBoostingClassifier

    # Initiating Gradient Boost Classifier
    clf = GradientBoostingClassifier()

    # Fitting Training Data into Model
    clf.fit(train_x,train_y)

    # Prediction output for Test Data
    predictions = clf.predict(test_x)

    return predictions

def apply_random_forest(train_x,train_y,test_x):
    """This Functions apply Random Forest on the following training dataset"""

    # Importing Classifier
    from sklearn.ensemble import RandomForestClassifier

    # Initiating Random Forest Classifier
    clf = RandomForestClassifier()

    # Fitting the Model
    clf.fit(train_x, train_y)

    # Predicting the results
    predictions = clf.predict(test_x)
    print("******* Predicting Values Using Random Forest ********")

    return predictions

def apply_adaboost(train_x,train_y,test_x):
    """ This Functions apply Ada Boost on the following training Data-Set """

    # Importing the pre-requisites
    print("******* Predicting Values Using Ada Boost ********")

    from sklearn import model_selection
    from sklearn.ensemble import AdaBoostClassifier
    num_trees = 30  # used for Adaboost
    seed = 7  # used for Adaboost
    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    # Initiating the classifier
    clf = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

    # Fitting the Training Data into Model
    clf.fit(train_x, train_y)

    # Finding Predictions for Test Data
    predictions = clf.predict(test_x)
    return predictions

def apply_decision_tree(train_x,train_y,test_x):
    """This Functions apply Decision Tree on the following training dataset"""
    print("******* Predicting Values Using Decision Tree Classifier ********")

    # Importing Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier

    # Initiating Decision Tree Classifier
    clf = DecisionTreeClassifier()

    # Fitting Training Data into Model
    clf.fit(train_x,train_y)

    # Prediction values for test data
    predictions = clf.predict(test_x)

    return predictions

def apply_xg_boost(train_x,train_y,test_x):
    """This Functions apply XG Boost on the following training dataset"""
    print("******* Predicting Values Using XG boost Classifier ********")

    # Importing XG Boost Classifier
    import xgboost as xgb

    # Initiating XG Boost Classifier
    # clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3, n_jobs=-1)
    clf = xgb.XGBClassifier()

    # Fitting Training Data into Model
    clf.fit(train_x, train_y)

    # Prediction Values for Test Data
    predictions = clf.predict(test_x)
    return predictions

def apply_lstm(train_x, train_y, test_x):
    """This Functions apply Long Short Term Memory RNN on the following training dataset"""

def split_x_and_y(data_frame, target="close", forecast_days = 1, single_prediction=True):
    print("Splitting X and Y Values")
    data_frame = data_frame.fillna(0)

    for name in list(data_frame.dtypes.index):
        # output[name] = list(data_frame[name].values)
        data_frame[name] = data_frame[name].apply(lambda x: int(x))

    forecast_out = forecast_days # predicting 30 days into future
    target_key = "Predicted_"+target
    data_frame[target_key] = data_frame[target]
    data_frame[target_key] = data_frame[target_key].apply(lambda x: int(x))
    data_frame[target_key] = data_frame[target_key].shift(-forecast_days)
    data_frame = data_frame[:-forecast_out]

    print("********* df ************")
    print(data_frame.tail())

    y_data = data_frame[target_key].values
    x_data = data_frame.drop([target_key], axis=1)


    # Splitting Randomized Data using train test split function
    if not single_prediction:
        train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.2)

    else:
        # Splitting Train X and Y data using Slicing Functions
        train_x = x_data[:-1]
        train_y = y_data[:-1]
        test_x = x_data.tail(1)
        test_y = y_data[-1]

    # print("X_train : {}, Y_Train : {} , X_test : {}, y_test : {} ".format(len(train_x),len(train_y),len(test_x),len(test_y)))
    return train_x, train_y, test_x, test_y

def split_x_and_y_final(data_frame, target="close", forecast_days = 1):
    print("Splitting X and Y Values")
    data_frame = data_frame.fillna(0)
    print("lenght of data frame : ",len(data_frame))
    for name in list(data_frame.dtypes.index):
        # output[name] = list(data_frame[name].values)
        data_frame[name] = data_frame[name].apply(lambda x: int(x))

    forecast_out = forecast_days # predicting n days into future

    target_prices = ["open", "high", "low", "close"]

    for price in target_prices:
        keyname = "Predicted_"+price
        data_frame[keyname] = data_frame[price]
        data_frame[keyname] = data_frame[keyname].shift(-forecast_days)

    # data_frame["Prediction"] = data_frame[target]
    # data_frame['Prediction'] = data_frame['Prediction'].apply(lambda x: int(x))
    # data_frame["Prediction"] = data_frame['Prediction'].shift(-forecast_days)

    # data_frame = data_frame[:-forecast_out]
    print("********* df ************")
    print(data_frame.tail())

    print("tailed : ",type(data_frame.iloc[-1]))

    # y_data = data_frame['Prediction'].values
    # x_data = data_frame.drop(['Prediction'], axis=1)
    #
    # train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.2)
    #
    # print("X_train : {}, Y_Train : {} , X_test : {}, y_test : {} ".format(len(train_x),len(train_y),len(test_x),len(test_y)))
    # return train_x, train_y, test_x, test_y

def movement_prediction(current_price, predicted_price):
    """ This Function is used to predict the price trend movement """
    if predicted_price<=current_price:
        return "BUY"
    else:
        return "SELL"

def run_prediction(train_x, train_y, test_x, test_y, target="close", apply_lr=True, apply_svr=False, apply_rf=False, apply_dt=False, apply_xgb=False, apply_adab=False, apply_lst=False, apply_gb=False, apply_linr=False):

    print("Running Predictions")
    output = {}
    output["buy_prices"]=[]
    output["sell_prices"]=[]
    output[target+"_Actual"] = test_y

    if apply_lr:
        predicted_price = target+"_pred_logreg"
        output[predicted_price] = apply_logistic_regression(train_x, train_y, test_x, test_y)[0]
        predicted_mov = target+"_logreg_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov]=="BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])

    if apply_linr:
        predicted_price = target + "_pred_linreg"
        output[predicted_price] = apply_linear_regression(train_x, train_y, test_x, test_y)[0]
        predicted_mov = target + "_linreg_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov]=="BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])

    if apply_svr:
        predicted_price = target + "_pred_svr"
        output[predicted_price] = apply_SVR(train_x, train_y, test_x)[0]
        predicted_mov = target + "_svr_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov] == "BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])

    if apply_dt:
        predicted_price = target + "_pred_dt"
        output[predicted_price] = apply_decision_tree(train_x, train_y, test_x)[0]
        predicted_mov = target + "_dt_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov]=="BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])

    if apply_rf:
        predicted_price = target + "_pred_rf"
        output[predicted_price] = apply_random_forest(train_x, train_y, test_x)[0]
        predicted_mov = target + "_rf_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov]=="BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])


    if apply_xgb:
        predicted_price = target + "_pred_xgb"
        output[predicted_price] = apply_xg_boost(train_x, train_y, test_x)[0]
        predicted_mov = target + "_xgb_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov]=="BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])

    if apply_adab:
        predicted_price = target + "_pred_adab"
        output[predicted_price] = apply_adaboost(train_x, train_y, test_x)[0]
        predicted_mov = target + "_adab_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov]=="BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])

    if apply_lst:
        predicted_price = target + "_pred_lstm"
        output[predicted_price] = apply_lstm(train_x, train_y, test_x)[0]
        predicted_mov = target + "_lstm_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov]=="BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])

    if apply_gb:
        predicted_price = target + "_pred_gb"
        output[predicted_price] = apply_gradient_boost(train_x, train_y, test_x)[0]
        predicted_mov = target + "_gb_mov"
        output[predicted_mov] = movement_prediction(test_y, output[predicted_price])
        if output[predicted_mov] == "BUY":
            output["buy_prices"].append(output[predicted_price])
        else:
            output["sell_prices"].append(output[predicted_price])
    # print("******** Output ***************")
    # final_output = pd.DataFrame(output)
    # print(final_output.head())
    # final_output.to_csv("strategy_three_with_all_ml.csv")
    return output

def prepare_features_data_st1(data_frame):
    """ This Function Prepares Training and Test Data for Strategy One.
     The Training Data includes Features like open, high, low, close, volume Prices Only """

    print("Preparing Training and Test Data for Strategy One ")
    data_frame = data_frame.drop("date", axis=1)
    data_frame = data_frame[['open', 'high',"low", "close"]]
    return data_frame

def prepare_features_data_st2(data_frame, target_price="close", sma = 50, ema = 50):
    """ This Function Prepares Training and Test Data for Strategy Two.
     The Training Data includes Features like open, High, Low, Close, Volume, Simple Moving Average, Exponential Moving Average"""

    # Selecting only specified columns
    data_frame = data_frame[['open', 'high', "low", "close"]]

    # Finding Simple Moving Average for user defined values
    data_frame['SMA'] = data_frame[target_price].rolling(sma).mean()

    # Finding Exponential Moving Average for user defined values
    data_frame["EMA"] = data_frame[target_price].ewm(span=ema,adjust=False).mean()

    # Filling Infinite or non integer value with Mean Average Values
    data_frame['SMA'] = data_frame['SMA'].fillna(data_frame['SMA'].mean())
    data_frame['EMA'] = data_frame['EMA'].fillna(data_frame['EMA'].mean())

    data_frame = data_frame.drop("date", axis=1)

    return data_frame

def prepare_features_data_st3(data_frame,target_price="close", sma=50, ema=50,forecast_days=1):
    """ This Function Prepares Training and Test Data for Strategy One
    The Training Data includes Features like
    Open, High, Low, Close, Volume,
    Standard Deviation, Simple Moving Average, Exponential Moving,
    Upper Bollinger Band, Lower Bollinger Band,Return Classifier """

    # Selecting Specific Columns
    data_frame = data_frame[['open', 'high', "low", "close"]]
    data_frame = data_frame.drop("date", axis=1)

    # Finding Simple Moving Average for user defined values
    data_frame['SMA'] = data_frame[target_price].rolling(sma).mean()

    # Finding Exponential Moving Average for user defined values
    data_frame["EMA"] = data_frame[target_price].ewm(span=ema, adjust=False).mean()

    # # Filling Infinite or non integer value with Mean Average Values
    data_frame['SMA'] = data_frame['SMA'].fillna(data_frame['SMA'].mean())
    data_frame['EMA'] = data_frame['EMA'].fillna(data_frame['EMA'].mean())

    # calculate daily returns
    data_frame['returns'] = np.log(data_frame[target_price] / data_frame[target_price].shift(forecast_days))
    data_frame['returns'] = data_frame['returns'].fillna(0)

    # data_frame['returns_1'] = data_frame['returns'].fillna(0)
    data_frame['returns'] = data_frame['returns'].replace([np.inf, -np.inf], np.nan)
    data_frame['returns'] = data_frame['returns'].fillna(0)

    # we apply the defined classifier above, being computeClassification,
    # to determine whether this will be 1 or -1 based on % move up or % move down on the daily return

    # data_frame['returns_final'] = data_frame['returns_2'].fillna(0)
    # print(data_frame['returns_final'])

    # SIGNALS
    data_frame['Stdev'] = data_frame[target_price].rolling(window=sma).std()  # calculate rolling std

    # Filling Infinite or non integer value with Mean Average Values
    data_frame['Stdev'] = data_frame['Stdev'].fillna(data_frame['Stdev'].mean())

    # Bollinger Bands
    data_frame['Upper Band'] = data_frame['SMA'] + (data_frame['Stdev'] * 1)  # 1 standard deviations above
    data_frame['Lower Band'] = data_frame['SMA'] - (data_frame['Stdev'] * 1)  # 1 standard deviations below

    data_frame['RSI'] = RSI(data_frame[target_price], 14)  # RSI function of series defined by the close price, and period of choosing (defaulted to 14)

    # # Momentum setup for parameters over a rolling mean time window of 2 ( i.e average of past two day returns)
    # data_frame['mom'] = np.sign(data_frame['returns'].rolling(2).mean())
    # #
    # # Compute the last column (Y) -1 = down, 1 = up by applying the defined classifier above to the 'returns' dataframe
    # data_frame.iloc[:, len(data_frame.columns) - 1] = data_frame.iloc[:, len(data_frame.columns) - 1].apply(computeClassification)

    return data_frame

def map_stock_to_industry(tickers_list):
    """ This Function is used to map stocks names to their Corresponding Industries"""
    print("****** Mapping Tickers to their Corresponding Industries *******")

    # Reading Nifty 500 Stocks Files
    nifty_list = pd.read_csv("./Data/ind_nifty500list.csv")

    # Slicing Dataframe for Desired Tickers only
    output = nifty_list.loc[nifty_list.Symbol.isin(tickers_list)]
    out={}

    # Making an dictionary object for output
    for i in range(len(output)):
        out[output["Symbol"].values[i]] = output["Industry"].values[i]

    return out

def find_nearest_price(actual_price, predicted_prices):
    """This Function is used to find the nearest Target Price from the Last/Actual Price"""
    return min(predicted_prices, key=lambda x: abs(x - actual_price))

def check_gann_place(close_price):
    gann_sheet = pd.read_excel("./Docs/Gann Angle Calculator.xlsx",sheetname="GANN Angles-45")
    gann_sheet = gann_sheet.drop(["ODD-1","ODD-2","ATR"], axis=1)
    keys_list = list(gann_sheet.keys())
    gann_sheet = gann_sheet.fillna(0)
    print("********** Reading GANN Sheet **************")
    print(gann_sheet.head())
    pos_values = []
    pos_indices=[]
    neg_values = []
    neg_indices=[]

    for key in keys_list:
        diff_list = gann_sheet[key].apply(lambda x : close_price - x)
        pos_val,neg_val, pos_index,neg_index = find_pos_and_neg(list(diff_list))
        pos_values.append(pos_val)
        neg_values.append(neg_val)
        neg_indices.append(neg_index)
        pos_indices.append(pos_index)

    pos_col_index = pos_values.index(min(pos_values))
    pos_val_index = pos_indices[pos_col_index]
    nearby_pos_col = keys_list[pos_col_index]
    nearby_pos_val = gann_sheet[nearby_pos_col][pos_val_index]

    neg_col_index = neg_values.index(max(neg_values))
    neg_val_index = neg_indices[neg_col_index]
    nearby_neg_col = keys_list[neg_col_index]
    nearby_neg_val = gann_sheet[nearby_neg_col][neg_val_index]

    print()
    print("nearby_pos_col  : ",nearby_pos_col)
    print("nearby_pos_val  : ",nearby_pos_val)
    print("nearby_pos_indx : ",pos_val_index)
    print()
    print("nearby_neg_col  : ",nearby_neg_col)
    print("nearby_neg_val  : ",nearby_neg_val)
    print("nearby_neg_indx : ",neg_val_index)

    return nearby_pos_col, nearby_pos_val, pos_val_index, nearby_neg_col, nearby_neg_val, neg_val_index

def find_pos_and_neg(integers_list):
    neg_val=-1000
    pos_val=1000
    pos_index=1000
    neg_index=1000
    try:
        neg_val = max([n for n in integers_list if n<0])
        neg_index = integers_list.index(neg_val)
    except:pass
    try:
        pos_val = min([n for n in integers_list if n>0])
        pos_index = integers_list.index(pos_val)
    except:pass
    return pos_val,neg_val,pos_index,neg_index

def group_by_trade_type(dataframe):
    """Function used to Group Stocks by Forecasted Trade Type"""
    print("******* Grouping Stocks by their Trade Types *********")
    print(dataframe.head())
    print("************ Finding Buy Sectors *********************")
    buy_df = dataframe[dataframe["final_flag"]=="BUY"]
    print("************ Finding Sell Sectors ********************")
    sell_df = dataframe[dataframe["final_flag"]=="SELL"]
    return buy_df, sell_df

def group_by_sectors(dataframe,trade_type):
    """Function used to Group Stocks by Industries"""
    _sectors = dataframe.groupby(["industry"])
    print("************* {} Sectors ****************".format(trade_type))
    for i in _sectors:
        filename = i[0] + "_"+trade_type+"_" + str(datetime.datetime.now().date()) + ".csv"
        out = pd.DataFrame(i[1])
        out = out.sort_values("return", ascending=False)
        # out.to_csv("./Daily_Predictions/" + filename)
        print(out)

def get_kite_inst_token(trading_symbol, database):
    "This Function is used to get kite instrument token from mongodb"
    token=0
    try:
        token =list(database.find({
                                    "exchange":"NSE",
                                    "instrument_type":"EQ",
                                    "tradingsymbol":trading_symbol},
                                {   "_id":0,
                                    "instrument_token":1}))[0]['instrument_token']
    except Exception as e:
        print("Exception in getting instrument_token : ",e)
        pass
    return token

def save_predictions_to_mongo(data_frame, data_base):
    print("******* Saving Daily Predictions to MongoDB ********")
    print(data_frame.head())
    output = []
    df = data_frame.get(['final_flag','ticker','industry','BUY_confidence','SELL_confidence','return'])
    for i in range(0,len(df)-1):
        out = {}
        out['timestamp']=datetime.datetime.now()
        out["prediction_date"] = (datetime.datetime.today()+timedelta(days=1)).date().isoformat()
        if df['final_flag'][i]=='BUY':
            out["confidence"]=df['BUY_confidence'][i]
        else:
            out["confidence"]=df['SELL_confidence'][i]
        out['industry']=df['industry'][i]
        out['trading_symbol']=df['ticker'][i]
        out['trade_type'] = df['final_flag'][i]
        out['instrument_token'] = get_kite_inst_token(df['ticker'][i], data_base['instruments'])
        out['return']= df['return'][i]
        data_base['predictions'].insert(out)

def read_predictions_from_mongo(data_base):
    print("******* Reading Daily Predictions from Mongo *******")
    d = (datetime.datetime.today()+timedelta(days=1)).date().isoformat()
    data_frame = pd.DataFrame(list(data_base.find({"prediction_date":d})))
    print(data_frame.head())
    return data_frame

def main_function(db, index="NIFTY500", database="mongo",**kwargs):
    training_strategies = ["st1"]
    target_prices = ["close"]

    if "training_strategies" in kwargs:
        training_strategies = kwargs["training_strategies"]

    if "targets" in kwargs:
        target_prices = kwargs["target_prices"]


    # Defining Blank DataFrame
    df = pd.DataFrame()

    if index=="NIFTY500":
        # Reading Stocks for Nifty 500 List
        df = pd.read_csv("./Data/ind_nifty500list.csv")
        print("********* Reading Stocks for Nifty 500 ***********")

    elif index == "NIFTY_MIDCAP_150":
        # Reading Stocks for Nifty MidCap 150
        print("********* Reading Stocks for Nifty MidCap 150 ***********")
        df = pd.read_csv("./Data/ind_nifty500list.csv")

    ## TODO : Add New Stock Index in Future As per the Advancements

    print(df.head())
    print("stocks_list : ",len(df))

    # Creating a List of Tickers Name
    tickers_list = list(df.Symbol.values)

    # Mapping Tickers Name to Respective Industries
    ticker_dict = map_stock_to_industry(tickers_list)

    final_output = []
    print("tickers_list : ",tickers_list)
    # Reading Tickers from tickers list and applying further process on them
    for ticker in tickers_list[:10]:
        print("*********** Reading Ticker : {} ************".format(ticker))

        final_out = {}
        final_out["ticker"] = ticker
        final_out["industry"] = ticker_dict[ticker]

        try:
            data_df = pd.DataFrame()
            if database=="csv":
                print("********* Reading {} Data from CSV Base ***********".format(ticker))
                # Reading End of Day Data for ticker from CSV File
                try:
                    data_df = get_eod_data_csv("./Data/"+ticker+".csv")
                except Exception as e:
                    print("Exception in reading data from csv : ",e)
                    pass

            elif database=="mongo":
                print("********* Reading {} Data from Mongo Base *********".format(ticker))
                # Reading End of Day Data for ticker from MongoDB
                try:
                    data_df = get_eod_data_mongodb(db, ticker)
                except Exception as e:
                    print("Exception in reading data from MongoDB : ",e)
                    pass

            print("------------- Data From Source ---------------")
            print(data_df.head())
            data_df = data_df.drop(['_id'],axis=1)
            print()
            for strategy in training_strategies:
                print("******** Preparing training data with {} Strategy **********".format(strategy))
                data_frame = pd.DataFrame()

                # Selecting Strategy
                if strategy=="st1":
                    final_out["training_strategy"] = "st1"
                    data_frame = prepare_features_data_st1(data_frame=data_df)
                elif strategy=="st2":
                    final_out["training_strategy"] = "st2"
                    data_frame = prepare_features_data_st2(data_frame=data_df)
                elif strategy=="st3":
                    final_out["training_strategy"] = "st3"
                    data_frame = prepare_features_data_st3(data_frame=data_df)

                print("************* Data frame with features ***************")
                print(data_frame.tail())

                # Defining the Target Prices to make predictions
                # target_prices = ["close","high","open","low"]

                for target in target_prices:
                    # Splitting Training and Test Data
                    train_x, train_y, test_x, test_y = split_x_and_y(data_frame, target=target, forecast_days=1)
                    out_dict = run_prediction(train_x, train_y, test_x, test_y,
                                              target=target,
                                              # apply_lr=True,
                                              apply_svr=True,
                                              apply_rf=True,
                                              apply_dt=True,
                                              # apply_linr=True,
                                              # apply_xgb=True,
                                              # apply_adab=True,
                                              # apply_gb=True,
                                              )

                    print("test_y = ",test_y)

                    # Checking if Buy Prices are there in Predictions
                    if out_dict["buy_prices"]:

                        # Finding Nearest Buy Prices to previous target price
                        out_dict["buy_price"] = find_nearest_price(test_y, out_dict["buy_prices"])

                        # Calculating Return Percentages
                        out_dict["return"] = (out_dict["buy_price"]-test_y)/test_y *100

                    # Checking if Sell Prices are there in Predictions
                    if out_dict["sell_prices"]:

                        # Finding Nearest Sell Price close to previous target price
                        out_dict["sell_price"] = find_nearest_price(test_y, out_dict["sell_prices"])

                        # Calculating Return Percentages
                        out_dict["return"] = (test_y-out_dict["sell_price"])/test_y *100

                    # Updating Prediction and Prices to Final Output
                    final_out.update(out_dict)

                # Finding total Buy and Sell
                total_count = list(final_out.values()).count("BUY")+list(final_out.values()).count("SELL")

                # Finding Buy and Sell Counts
                buy_count = list(final_out.values()).count("BUY")
                sell_count = list(final_out.values()).count("SELL")
                final_out["buy_count"] = buy_count
                final_out["sell_count"] = sell_count
                final_out["BUY_confidence"] = buy_count/total_count*100
                final_out["SELL_confidence"] = sell_count/total_count*100

                # Finding Dominancy of Buy and Sell Count
                if final_out["BUY_confidence"] >= 50:
                    final_out["final_flag"] = "BUY"
                else:
                    final_out["final_flag"] = "SELL"
                final_output.append(final_out)

        except Exception as e:
            print("Exception in start following ",e)
            pass

    fin_df = pd.DataFrame(final_output)

    # fin_df = fin_df.drop(["sell_prices","buy_prices","buy_count","sell_count"], axis=1)
    fin_df = fin_df.drop(["sell_prices","buy_prices"], axis=1)

    print("********* Final Prediction DataFrame ************")
    print(fin_df.head())
    return fin_df


if __name__ == '__main__':
    from pymongo import MongoClient
    db = MongoClient("localhost",27017)['hazamin']['instruments']
    main_db = MongoClient("localhost",27017)['hazamin']

    predicted_df = main_function(main_db,database="mongo")
    print("*********** predicted_df ***********")
    print(predicted_df)

    # Writing Intermediatory Output to CSV File
    # predicted_df.to_csv("nifty_500_predictions.csv")

    print("*********** Adding Predictions to MongoDB ***********")
    save_predictions_to_mongo(predicted_df, main_db)

    # instrument_tokens = []
    # tickers_list = ['3MINDIA', '8KMILES', 'ABB', 'ACC', 'AIAENG', 'APLAPOLLO', 'AUBANK', 'AARTIIND', 'ABAN', 'ADANIPORTS', 'ADANIPOWER', 'ADANITRANS', 'ABCAPITAL', 'ABFRL', 'ADVENZYMES', 'AEGISCHEM', 'AJANTPHARM', 'AKZOINDIA', 'APLLTD', 'ALKEM', 'ALBK', 'ALLCARGO', 'AMARAJABAT', 'AMBUJACEM', 'ANDHRABANK', 'APOLLOHOSP', 'APOLLOTYRE', 'ASAHIINDIA', 'ASHOKLEY', 'ASHOKA', 'ASIANPAINT', 'ASTERDM', 'ASTRAL', 'ATUL', 'AUROPHARMA', 'AVANTIFEED', 'DMART', 'AXISBANK', 'BASF', 'BEML', 'BFUTILITIE', 'BGRENERGY', 'BSE', 'BAJAJ-AUTO', 'BAJAJCORP', 'BAJAJELEC', 'BAJFINANCE', 'BAJAJFINSV', 'BAJAJHIND', 'BAJAJHLDNG', 'BALKRISIND', 'BALLARPUR', 'BALMLAWRIE', 'BALRAMCHIN', 'BANKBARODA', 'BANKINDIA', 'BATAINDIA', 'BERGEPAINT', 'BEL', 'BHARATFIN', 'BHARATFORG', 'BHEL', 'BPCL', 'BHARTIARTL', 'INFRATEL', 'BHUSANSTL', 'BIOCON', 'BIRLACORPN', 'BLISSGVS', 'BLUEDART', 'BLUESTARCO', 'BBTC', 'BOMDYEING', 'BRFL', 'BOSCHLTD', 'BRIGADE', 'BRITANNIA', 'CARERATING', 'CCL', 'CGPOWER', 'CRISIL', 'CADILAHC', 'CANFINHOME', 'CANBK', 'CAPF', 'CAPLIPOINT', 'CARBORUNIV', 'CASTROLIND', 'CEATLTD', 'CENTRALBK', 'CDSL', 'CENTURYPLY', 'CENTURYTEX', 'CERA', 'CHAMBLFERT', 'CHENNPETRO', 'CHOLAFIN', 'CIPLA', 'CUB', 'COALINDIA', 'COCHINSHIP', 'COFFEEDAY', 'COLPAL', 'CONCOR', 'COROMANDEL', 'CROMPTON', 'CUMMINSIND', 'CYIENT', 'DBREALTY', 'DBCORP', 'DCBBANK', 'DCMSHRIRAM', 'DLF', 'DABUR', 'DEEPAKFERT', 'DELTACORP', 'DEN', 'DENABANK', 'DHFL', 'DBL', 'DISHTV', 'DCAL', 'DIVISLAB', 'DIXON', 'LALPATHLAB', 'DRREDDY', 'DREDGECORP', 'EIDPARRY', 'EIHOTEL', 'EDELWEISS', 'EICHERMOT', 'EMAMILTD', 'ENDURANCE', 'ENGINERSIN', 'EQUITAS', 'ERIS', 'EROSMEDIA', 'ESCORTS', 'EVEREADY', 'EXIDEIND', 'FEDERALBNK', 'FINCABLES', 'FINPIPE', 'FSL', 'FCONSUMER', 'FLFL', 'FRETAIL', 'GAIL', 'GEPIL', 'GET&D', 'GHCL', 'GMRINFRA', 'GVKPIL', 'GDL', 'GATI', 'GICRE', 'GILLETTE', 'GSKCONS', 'GLAXO', 'GLENMARK', 'GODFRYPHLP', 'GODREJAGRO', 'GODREJCP', 'GODREJIND', 'GODREJPROP', 'GRANULES', 'GRAPHITE', 'GRASIM', 'GESHIP', 'GREAVESCOT', 'GREENPLY', 'GRUH', 'GUJALKALI', 'GUJFLUORO', 'GUJGASLTD', 'GMDCLTD', 'GNFC', 'GPPL', 'GSFC', 'GSPL', 'GULFOILLUB', 'HEG', 'HCL-INSYS', 'HCLTECH', 'HDFCBANK', 'HSIL', 'HATHWAY', 'HATSUN', 'HAVELLS', 'HEIDELBERG', 'HERITGFOOD', 'HEROMOTOCO', 'HEXAWARE', 'HFCL', 'HSCL', 'HIMATSEIDE', 'HINDALCO', 'HCC', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC', 'HONAUT', 'HUDCO', 'HDFC', 'HDIL', 'ITC', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDBI', 'IDFCBANK', 'IDFC', 'IFBIND', 'IFCI', 'IIFL', 'IL&FSTRANS', 'IRB', 'ITDCEM', 'ITI', 'IDEA', 'INDIACEM', 'ITDC', 'IBULHSGFIN', 'IBREALEST', 'IBVENTURES', 'INDIANB', 'IEX', 'INDHOTEL', 'IOC', 'IOB', 'ICIL', 'INDOCO', 'IGL', 'INDUSINDBK', 'INFIBEAM', 'NAUKRI', 'INFY', 'INOXLEISUR', 'INOXWIND', 'INTELLECT', 'INDIGO', 'IPCALAB', 'JBCHEPHARM', 'JKCEMENT', 'JKIL', 'JBFIND', 'JKLAKSHMI', 'JKTYRE', 'JMFINANCIL', 'JSWENERGY', 'JSWSTEEL', 'JAGRAN', 'JAICORPLTD', 'JISLJALEQS', 'JPASSOCIAT', 'JPPOWER', 'J&KBANK', 'JETAIRWAYS', 'JINDALPOLY', 'JINDALSAW', 'JSLHISAR', 'JSL', 'JINDALSTEL', 'JCHAC', 'JUBLFOOD', 'JUBILANT', 'JUSTDIAL', 'JYOTHYLAB', 'KPRMILL', 'KIOCL', 'KNRCON', 'KPIT', 'KRBL', 'KAJARIACER', 'KALPATPOWR', 'KANSAINER', 'KTKBANK', 'KARURVYSYA', 'KSCL', 'KEC', 'KESORAMIND', 'KITEX', 'KOLTEPATIL', 'KOTAKBANK', 'KWALITY', 'L&TFH', 'LTTS', 'LICHSGFIN', 'LAXMIMACH', 'LAKSHVILAS', 'LTI', 'LT', 'LAURUSLABS', 'LUPIN', 'MASFIN', 'MMTC', 'MOIL', 'MRF', 'MAGMA', 'MGL', 'MTNL', 'M&MFIN', 'M&M', 'MAHINDCIE', 'MHRIL', 'MANAPPURAM', 'MRPL', 'MANPASAND', 'MARICO', 'MARKSANS', 'MARUTI', 'MFSL', 'MAXINDIA', 'MCLEODRUSS', 'MERCK', 'MINDTREE', 'MINDACORP', 'MINDAIND', 'MOTHERSUMI', 'MOTILALOFS', 'MPHASIS', 'MUTHOOTFIN', 'NATCOPHARM', 'NBCC', 'NCC', 'NESCO', 'NHPC', 'NIITTECH', 'NLCINDIA', 'NMDC', 'NTPC', 'NH', 'NATIONALUM', 'NFL', 'NBVENTURES', 'NAVINFLUOR', 'NAVKARCORP', 'NETWORK18', 'NILKAMAL', 'OBEROIRLTY', 'ONGC', 'OIL', 'OMAXE', 'OFSS', 'ORIENTCEM', 'ORIENTBANK', 'PCJEWELLER', 'PIIND', 'PNBHOUSING', 'PNCINFRA', 'PFS', 'PTC', 'PVR', 'PAGEIND', 'PARAGMILK', 'PERSISTENT', 'PETRONET', 'PFIZER', 'PHOENIXLTD', 'PIDILITIND', 'PEL', 'PFC', 'POWERGRID', 'PRAJIND', 'PRESTIGE', 'PRSMJOHNSN', 'PGHH', 'PNB', 'QUESS', 'RBLBANK', 'RADICO', 'RAIN', 'RAJESHEXPO', 'RALLIS', 'RAMCOSYS', 'RKFORGE', 'RCF', 'RTNPOWER', 'RAYMOND', 'REDINGTON', 'RELAXO', 'RELCAPITAL', 'RCOM', 'RHFL', 'RELIANCE', 'RELINFRA', 'RNAVAL', 'RPOWER', 'RELIGARE', 'REPCOHOME', 'RUPA', 'RECLTD', 'SHK', 'SBILIFE', 'SJVN', 'SKFINDIA', 'SMLISUZU', 'SREINFRA', 'SRF', 'SADBHAV', 'SADBHIN', 'SANOFI', 'SCHAEFFLER', 'SIS', 'SHANKARA', 'SFL', 'SCI', 'SHOPERSTOP', 'SHREECEM', 'RENUKA', 'SHRIRAMCIT', 'SRTRANSFIN', 'SIEMENS', 'SPTL', 'SOBHA', 'SOLARINDS', 'SONATSOFTW', 'SOUTHBANK', 'STARCEMENT', 'SBIN', 'SAIL', 'STRTECH', 'SUDARSCHEM', 'SPARC', 'SUNPHARMA', 'SUNTV', 'SUNDRMFAST', 'SUNTECK', 'SUPREMEIND', 'SUVEN', 'SUZLON', 'SWANENERGY', 'SYMPHONY', 'SYNDIBANK', 'SYNGENE', 'TIFIN', 'TTKPRESTIG', 'TVTODAY', 'TV18BRDCST', 'TVSMOTOR', 'TVSSRICHAK', 'TAKE', 'TNPL', 'TATACHEM', 'TATACOFFEE', 'TCS', 'TATAELXSI', 'TATAGLOBAL', 'TATAINVEST', 'TATAMTRDVR', 'TATAMOTORS', 'TATAPOWER', 'TATASPONGE', 'TATASTEEL', 'TECHM', 'TECHNO', 'TEXRAIL', 'RAMCOCEM', 'THERMAX', 'THOMASCOOK', 'THYROCARE', 'TIMETECHNO', 'TIMKEN', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TRENT', 'TRIDENT', 'TIINDIA', 'UCOBANK', 'UFLEX', 'UPL', 'UJJIVAN', 'ULTRACEMCO', 'UNICHEMLAB', 'UNIONBANK', 'UNITECH', 'UBL', 'MCDOWELL-N', 'VGUARD', 'VIPIND', 'VRLLOG', 'WABAG', 'VAKRANGEE', 'VTL', 'VBL', 'VEDL', 'VIJAYABANK', 'VINATIORGA', 'VOLTAS', 'WABCOINDIA', 'WELCORP', 'WELSPUNIND', 'WHIRLPOOL', 'WIPRO', 'WOCKPHARMA', 'YESBANK', 'ZEEL', 'ZEELEARN', 'ECLERX']
    # instrument_tokens = [121345, 3329, 5633, 3350017, 6599681, 5436929, 1793, 2561, 3861249, 4451329, 5533185, 7707649, 4617985, 10241, 2079745, 375553, 6483969, 2995969, 2760193, 3456257, 25601, 325121, 2524673, 40193, 41729, 1376769, 54273, 5166593, 60417, 386049, 3691009, 67329, 70401, 2031617, 5097729, 1510401, 94209, 101121, 3729153, 3888385, 5013761, 4267265, 4999937, 3848705, 81153, 4268801, 78849, 78081, 85761, 3937025, 86529, 87297, 1195009, 1214721, 94977, 103425, 98049, 4995329, 108033, 112129, 134657, 2714625, 7458561, 114433, 2911489, 122881, 4931841, 126721, 2127617, 97281, 558337, 3887105, 140033, 7452929, 2931713, 194561, 193793, 2029825, 149249, 2763265, 3903745, 999937, 152321, 320001, 3905025, 3812865, 5420545, 3406081, 160001, 3849985, 163073, 524545, 175361, 177665, 1459457, 5215745, 5506049, 2858241, 3876097, 1215745, 189185, 4376065, 486657, 1471489, 4639745, 4577537, 3513601, 207617, 3771393, 197633, 211713, 3851265, 4536833, 1187329, 215553, 4630017, 3721473, 5556225, 2800641, 5552641, 2983425, 225537, 2885377, 234497, 235265, 3870465, 232961, 3460353, 4818433, 1256193, 4314113, 5415425, 5140481, 245249, 3016193, 173057, 261889, 265729, 266497, 3661825, 7689729, 7872513, 4704769, 1207553, 2012673, 4296449, 288513, 3463169, 3385857, 3004161, 3504129, 70913, 403457, 821761, 295169, 1895937, 302337, 36865, 2585345, 2796801, 4576001, 3039233, 151553, 315393, 3526657, 316161, 1020673, 2957569, 324353, 329985, 2713345, 1332225, 300545, 5051137, 319233, 3378433, 1124097, 342017, 339713, 1850625, 341249, 361473, 4647425, 996353, 2513665, 592897, 1177089, 345089, 2747905, 5619457, 3669505, 348161, 348929, 352001, 4592385, 359937, 356865, 364545, 874753, 5331201, 340481, 3789569, 424961, 1270529, 5573121, 4774913, 377857, 2863105, 3060993, 380161, 381697, 3023105, 4696321, 3920129, 1439233, 428801, 3677697, 387841, 4940545, 7712001, 3699201, 3663105, 56321, 387073, 415745, 2393089, 3068673, 2989313, 2883073, 1346049, 4159745, 3520257, 408065, 3384577, 2010113, 1517057, 2865921, 418049, 441857, 3397121, 3908097, 3453697, 3695361, 3491073, 4574465, 3001089, 3382017, 1316609, 2661633, 2933761, 3011329, 1442049, 2997505, 449537, 774145, 3149825, 2876417, 1723649, 1149697, 4632577, 931073, 7670273, 3877377, 3817473, 4896257, 3912449, 1790465, 2707713, 462849, 464385, 306177, 2061825, 470529, 3832833, 3394561, 475905, 7398145, 3871745, 492033, 4581633, 6386689, 4752385, 511233, 506625, 504321, 4561409, 2939649, 4923905, 2672641, 50945, 4596993, 5332481, 582913, 2919169, 4488705, 587265, 3400961, 519937, 3823873, 4437249, 4879617, 584449, 2560513, 1041153, 2708225, 2815745, 548353, 4542721, 3057409, 240641, 3675137, 6629633, 3623425, 1076225, 3826433, 1152769, 6054401, 1003009, 8042241, 593665, 3944705, 4454401, 2955009, 2197761, 3924993, 2977281, 3031041, 1629185, 3564801, 1027585, 3756033, 2702593, 3612417, 619777, 5181953, 633601, 4464129, 3802369, 2748929, 7702785, 636673, 7455745, 6191105, 4840449, 2402561, 5786113, 2906881, 3365633, 3689729, 4385281, 4701441, 2905857, 676609, 3725313, 681985, 617473, 3660545, 3834113, 692481, 5197313, 701185, 648961, 2730497, 4532225, 4708097, 2813441, 1894657, 720897, 2009857, 2921217, 733697, 4485121, 731905, 3649281, 6201601, 737793, 3375873, 5563649, 738561, 141569, 4465665, 3906305, 3857409, 7577089, 6585345, 3930881, 2870273, 5582849, 4834049, 815617, 867073, 836353, 837889, 3388417, 2718209, 369153, 258817, 5504257, 5202177, 4911105, 780289, 3024129, 794369, 3078657, 3005185, 1102337, 806401, 5494273, 3539457, 3412993, 1688577, 1522689, 5399297, 779521, 758529, 2383105, 851713, 3785729, 857857, 3431425, 856321, 4516097, 860929, 2875649, 3076609, 6936321, 6192641, 1837825, 2622209, 5565441, 907777, 2886401, 3637249, 2170625, 3646721, 3818753, 1018881, 871681, 185345, 2953217, 873217, 878593, 414977, 4343041, 884737, 877057, 419585, 895745, 3465729, 5587969, 523009, 889601, 891137, 4360193, 3764993, 3634689, 897537, 900609, 3529217, 502785, 2479361, 79873, 2873089, 269569, 2889473, 4369665, 2952193, 916225, 2752769, 920065, 4278529, 2674433, 3932673, 947969, 2226177, 5168129, 3415553, 530689, 4843777, 784129, 2426625, 4445185, 951809, 4330241, 3026177, 2880769, 4610817, 969473, 1921537, 3050241, 975873, 5338113, 3885825]
    # for ticker in tickers_list:
    #
    #     instrument = get_kite_inst_token(ticker,db)
    #     if instrument:
    #         instrument_tokens.append(instrument)
    #     else:
    #         print("instrument token not found")
    # print("instrument_tokens : ",len(instrument_tokens))
