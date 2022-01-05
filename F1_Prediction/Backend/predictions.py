import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import make_interp_spline
import plotly.express as px
import plotly.graph_objects as go

data = None
df = None
data_complete = None
def dataCreation(data):
    global df
    df_race = data[data['sessionName'] == 'Race']

    df_practice1 = data[data['sessionName'] == 'Practice 1'][['fastestLapRank','fastestLapTime','year','gpName','driverId']]
    df_practice2 = data[data['sessionName'] == 'Practice 2'][['fastestLapRank','fastestLapTime','year','gpName','driverId']]
    df_practice3 = data[data['sessionName'] == 'Practice 3'][['fastestLapRank','fastestLapTime','year','gpName','driverId']]

    df_practice1 = df_practice1.rename(columns={"fastestLapRank": "fastestLapRankP1","fastestLapTime": "fastestLapTimeP1"})
    df_practice2 = df_practice2.rename(columns={"fastestLapRank": "fastestLapRankP2","fastestLapTime": "fastestLapTimeP2"})
    df_practice3 = df_practice3.rename(columns={"fastestLapRank": "fastestLapRankP3","fastestLapTime": "fastestLapTimeP3"})

    df_quali = data[data['sessionName'] == 'Qualifying'][['Q1','Q2','Q3','year','gpName','driverId']]
    df_race = df_race.drop(['Q1','Q2','Q3'], axis=1)

    df_race['totalTime'] = df_race['totalTime'].fillna(df_race['totalTime'].max())
    
    race_quali = pd.merge(df_race, df_quali, how="left", on=['year','gpName','driverId'])
    race_quali_1 = pd.merge(race_quali, df_practice1, how="left", on=['year','gpName','driverId'])
    race_quali_12 = pd.merge(race_quali_1, df_practice2, how="left", on=['year','gpName','driverId'])
    global data_complete
    data_complete = pd.merge(race_quali_12, df_practice3, how="left", on=['year','gpName','driverId'])
    df_pred_postion = data_complete.drop(['positionText','points','laps','status','fastestLapNumber', 'fastestLapRank', 'fastestLapAvgSpeed',
        'fastestLapTime', 'totalTime', 'TimeInterval','code','driverId','sessionName'], axis=1)

    timeStrToInt(df_pred_postion,'fastestLapTimeP1')
    timeStrToInt(df_pred_postion,'fastestLapTimeP2')
    timeStrToInt(df_pred_postion,'fastestLapTimeP3')

    timeStrToIntQ1(df_pred_postion)
    timeStrToIntQ2_3(df_pred_postion,2)
    timeStrToIntQ2_3(df_pred_postion,3)

    df_pred_postion = df_pred_postion.dropna(subset=['position'])

    df_pred_postion.loc[df_pred_postion.fastestLapRankP1.isna(), 'fastestLapRankP1'] = 21
    df_pred_postion.loc[df_pred_postion.fastestLapRankP2.isna(), 'fastestLapRankP2'] = 21
    df_pred_postion.loc[df_pred_postion.fastestLapRankP3.isna(), 'fastestLapRankP3'] = 21

    df_pred_postion.loc[df_pred_postion.grid == 0.0, 'grid'] = 21.0

    df_features2 =  pd.get_dummies(df_pred_postion, columns = ['constructorId'])
    df_features2=df_features2.drop(['DriverNationality','constructorNationality','gpName'], axis=1)

    scaler = MinMaxScaler()
    to_scale = ['Q1','Q2', 'Q3','fastestLapTimeP1',
        'fastestLapTimeP2', 'fastestLapTimeP3']
    df_scaled2 = df_features2.copy()
    df_scaled2[to_scale] = scaler.fit_transform(df_scaled2[to_scale].to_numpy())
    df_scaled2 = pd.DataFrame(df_scaled2, columns= df_features2.columns)
    df = df_scaled2
    return df_scaled2

def model_prevision_race(year,GpNumber):
    X_train,y_train,X_test,y_test = test_train_creation_gp(df,year,GpNumber)
    random_forest = RandomForestRegressor(random_state = 47) 
    model_rd = random_forest.fit(X_train, y_train)
    y_pred_noRank,acc_NoRank = evaluateNoDisplay(model_rd, X_test,y_test)
    y_pred,acc = evaluaterankNoDisplay( y_test, y_pred_noRank)
    return model_rd,y_pred,acc,X_test,X_train
def model_prevision_raceV2(year,GpNumber,n):
    X_train,y_train,X_test,y_test = test_train_creation_gpV2(df,year,GpNumber,n)
    random_forest = RandomForestRegressor(random_state = 47) 
    model_rd = random_forest.fit(X_train, y_train)
    y_pred_noRank,acc_NoRank = evaluateNoDisplay(model_rd, X_test,y_test)
    y_pred,acc = evaluaterankNoDisplay( y_test, y_pred_noRank)
    return model_rd,y_pred,acc,X_test,X_train

def timeStrToInt(df,columnName):
    l = []
    is_na = df[columnName].isna()
    for i in range(len(df[columnName])):
        is_nai = is_na[i]
        if  is_nai == False :
            value = df[columnName][i]
            if type(value) == np.int64:
                l.append(value)
            else:
                a = value.split(':')
                b = list(map(float,a[1:]))
                l.append(int((b[0]*60*1000 + b[1])*1000))
        else:
            l.append(58276750)
    df[columnName] = l

def timeStrToIntQ1(df):
    l = []
    is_na = df['Q1'].isna()
    for i in range(len(df['Q1'])):
        is_nai = is_na[i]
        if  is_nai == False :
            value = df['Q1'][i]
            if type(value) == np.int64:
                l.append(value)
            else:
                a = value.split(':')
                b = list(map(float,a))
                l.append(int((b[0]*60*1000 + b[1])*1000))
        else:
            l.append(120004382)
    df['Q1'] = l

def timeStrToIntQ2_3(df, Q_number):
    l = []
    Q = 'Q'+str(Q_number)
    is_na = df[Q].isna()
    for i in range(len(df[Q])):
        is_nai = is_na[i]
        if  is_nai == False :
            value = df[Q][i]
            if type(value) == np.int64:
                l.append(value)
            else:
                a = value.split(':')
                b = list(map(float,a))
                l.append(int((b[0]*60*1000 + b[1])*1000))
        else:
            c_name = 'Q'+ str((Q_number-1))
            l.append(df[c_name].iloc[i])
    df[Q] = l

def which_gp(gpNumber,year,df_):
    if gpNumber != 1:
        idx = df_[(df_['gpNumber']==gpNumber) & (df_['year']==year)].index[0]
        idx_begin = df_[df_['year']==year].index[0]
        idx_end = idx+20
        train = df_.iloc[idx_begin:idx]
        test = df_.iloc[idx:idx_end]
    else:
        idx = df_[(df_['gpNumber']==gpNumber) & (df_['year']==year)].index[0]
        idx_begin = df_[df_['year']==(year-1)].index[0]
        idx_end = idx+20
        train = df_.iloc[idx_begin:idx]
        test = df_.iloc[idx:idx_end]
    return train,test
def which_gpV2(gpNumber,year,df,n):
    if year != 2018:
        test,train = None,None
        if (gpNumber-n)<=0:
            gpy_1 = df[df['year']==year-1]['gpNumber'].max()
            idx = df[(df['gpNumber']==(gpNumber)) & (df['year']==year)].index[0]
            idx_begin = df[(df['gpNumber']==(gpy_1+(gpNumber-n))) & (df['year']==(year-1))].index[0]
            idx_end = idx+20
            train = df.iloc[idx_begin:idx]
            test = df.iloc[idx:idx_end]
        else:
            idx = df[(df['gpNumber']==(gpNumber)) & (df['year']==year)].index[0]
            idx_begin = df[df['year']==year].index[0]
            idx_end = idx+20
            train = df.iloc[idx_begin:idx]
            test = df.iloc[idx:idx_end]
    else:
        if gpNumber != 1:
            idx = df[(df['gpNumber']==gpNumber) & (df['year']==year)].index[0]
            idx_begin = df[df['year']==year].index[0]
            idx_end = idx+20
            train = df.iloc[idx_begin:idx]
            test = df.iloc[idx:idx_end]
    return train,test

def test_train_creation_gp(df,year,gpNumber):
    train, test = which_gp(gpNumber,year,df)
    X_train = train.drop(['position'], axis=1)
    y_train = train['position']
    X_test = test.drop(['position'], axis=1)
    y_test = test['position']
    return X_train,y_train,X_test,y_test
def test_train_creation_gpV2(df,year,gpNumber,n):
    train, test = which_gpV2(gpNumber,year,df,n)
    X_train = train.drop(['position'], axis=1)
    y_train = train['position']
    X_test = test.drop(['position'], axis=1)
    y_test = test['position']
    return X_train,y_train,X_test,y_test

def evaluateNoDisplay(model, test_features, test_labels):
    y_pred = model.predict(test_features) 
    error = abs(y_pred - test_labels)
    return y_pred,error_calculationNoDisplay(error,test_labels)

def evaluaterankNoDisplay(test_labels,y_pred):
    y_pred = rank(y_pred)
    error = abs(y_pred - test_labels)
    return y_pred,error_calculationNoDisplay(error,test_labels)

def error_calculationNoDisplay(error,y_test):
    round_error = round(error)
    mape = 100 * (error / y_test)
    count = 0
    for elmt in round_error:
        if elmt==0:
            count+=1
    return [round(np.mean(error), 2),round(np.mean(mape), 2),count/len(round_error)*100]

def rank(y_pred):
    y_pred_1 =  sorted(y_pred.copy())
    for i in range(len(y_pred_1)):
        for j in range(len(y_pred)):
            if y_pred_1[i] == y_pred[j]:
                y_pred_1[i] = j+1
    return y_pred_1

def data_driver(X_test):
    driver = data_complete['driverId'].iloc[(X_test.index[0]):X_test.index[0]+20]
    return list(driver)

def get_name(X_test):
    name = data_complete['gpName'].iloc[(X_test.index[0])]
    return name
def interval_acc(reg_scaled,interval):
    sum_rank = 0
    for i in reg_scaled:
        t = i[4]
        c = 0
        if len(t) == 20 :
            for pos in range(0,20):
                for j in range(0,(interval+1)):
                    if ((1+pos+j)==t[0][pos])|((1+pos-j)==t[0][pos]):
                        c+=1
                        break
            sum_rank+=((c/20)*100)
        else : 
            for pos in range(0,19):
                for j in range(0,(interval+1)):
                    if ((1+pos+j)==t[0][pos])|((1+pos-j)==t[0][pos]):
                        c+=1
                        break
            sum_rank+=((c/19)*100)
    return sum_rank/len(reg_scaled)
def position_acc(reg_scaled,position):
    sum_rank = 0
    for i in reg_scaled:
        t = i[4]
        c = 0  
        if t != reg_scaled[-1][4] :
            if (position==t[0][position-1]):
                c+=1

            sum_rank+=c
        else : 
            position -=1
            if (position==round(t[0][position-1])):
                c=c

            sum_rank+=c
    return (sum_rank/len(reg_scaled))*100
def evolution_acc(reg_scaled,gpNumber):
    sum_y = 0
    for y in [2018,2019,2020,2021]:
        sum_rank = 0
        gpn =0
        for i in reg_scaled:
            if (i[0][0] == y)&(i[0][1]<=gpNumber):
                t = i[4]
                c = 0
                gpn+=1
                if len(t) == 20 :
                    for pos in range(0,20):
                        if (1+pos)==t[0][pos]:
                            c+=1
                    sum_rank+=((c/20)*100)
                else:
                    for pos in range(0,19):
                        if (1+pos)==round(t[0][pos]):
                            c+=1
                    sum_rank+=((c/19)*100)
        if gpn!=0:
            sum_y+=sum_rank/gpn
    return sum_y/4
def test_deepness_Unique(n):

    year = [2018,2019,2020,2021]
    GpNumber = [i for i in range(1,23)]
    reg_scaled=[]

    for y in year:
        for g in GpNumber:
            try:
                X_train,y_train,X_test,y_test = test_train_creation_gpV2(df,y,g,n)


                random_forest = RandomForestRegressor(random_state = 47) 
                model_rd = random_forest.fit(X_train, y_train)
                y_pred,acc = evaluateNoDisplay(random_forest, X_test, y_test)
                l=[]
                l.append((y,g))
                l.append(acc)
                l.append(y_test)
                l.append(y_pred.copy())
                y_pred_2 = evaluaterankNoDisplay(y_test,y_pred)
                l.append(y_pred_2)
                reg_scaled.append(l)
            except:
                error = 0

    figures = []
    intervals = [interval_acc(reg_scaled,x) for x in range(15)]
    x = [x for x in range(15)]
    a = pd.DataFrame(x, columns = ['position Interval'])
    a['Mean accuracy in %'] = intervals
    a['Accuracy'] = [round(j,2) for j in intervals]
    a['c'] = ["Performance" for x in range(15)]
    fig = px.line(  a,x='position Interval', y='Mean accuracy in %', text='Accuracy',template = 'plotly_dark',color = 'c',color_discrete_sequence = ['darkred'])
    fig.update_traces(textposition='top center')
    fig.update_layout(title = "Accuracy in fonction of the position Interval")
    figures.append(fig.to_html(full_html = False))

    x = [x for x in range(1,21)]
    positions_acc = [position_acc(reg_scaled,x) for x in range(1,21)]
    a = pd.DataFrame(x, columns = ['position'])
    a['Mean accuracy in %'] = positions_acc
    a['Accuracy'] = [round(j,2) for j in positions_acc]
    a['c'] = ["Performance" for x in range(1,21)]

    fig = px.bar(  a,x='position', y='Mean accuracy in %', text='Accuracy',template = 'plotly_dark',color = 'c',color_discrete_sequence = ['darkred'])
    fig.update_layout(title = "Accuracy in fonction of the position")
    figures.append(fig.to_html(full_html = False))

    evolutions = [evolution_acc(reg_scaled,x) for x in range(1,23)]
    x = np.array([x for x in range(1,23)])
    X_Y_Spline = make_interp_spline(x, evolutions)

    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)

    a = pd.DataFrame(X_, columns = ['X_'])
    #a['Mean accuracy in %'] = evolutions
    acc_ = [round(j,2) for j in evolutions]
    acc= []
    ca = 0
    for i in range(500):
        if i%23 == 0 & i!= 500:
            
            acc.append(acc_[ca])
            ca+=1
        else:
            acc.append(None)
    a['c'] = ["Performance" for x in range(500)]
    a['Grand Prix Number'] = X_
    a['Mean accuracy in %'] = Y_
    a['Accuracy'] = acc
    fig = px.line(  a,x='Grand Prix Number', y='Mean accuracy in %',template = 'plotly_dark',text='Accuracy',color = 'c',color_discrete_sequence = ['darkred'])
    fig.update_traces(textposition='middle center')
    fig.update_layout(title = "Accuracy evolution during the season")
    figures.append(fig.to_html(full_html = False))

    return figures 

def init_from_local():
    global df,data
    data = pd.read_csv(r'..\Data\allData.csv')
    df = dataCreation(data)
    return df,data