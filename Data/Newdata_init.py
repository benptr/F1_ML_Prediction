import fastf1
from fastf1.core import Laps
from fastf1.livetiming.data import LiveTimingData
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache("C:/Users/lmbfr/OneDrive/Documents/Travail ESILV/Cours + Exercices/Semestre 7/Python Data Analysis/F1")

#df_global creation

session_letter = ['Q', 'R']
sessions = []
venues = {}
for year in range(2018,2022):
    for gp in range(1,23):
        for l in session_letter:
            try:
                session = fastf1.get_session(year, gp, l)
                sessions.append(session)
                
                venueKey = (year, session.weekend.name)
                if venueKey not in venues.keys():
                    venues[venueKey] = gp
                
            except:
                print("problem on",year, gp, l)

l=[]
year = []
gpNames = []
gpNumbers = []
label_session = []

for elmt in sessions:
    for res in elmt.results:
        l.append(res)
        year.append(elmt.weekend.year)
        gpNames.append(elmt.weekend.name)
        gpNumbers.append(venues[(elmt.weekend.year, elmt.weekend.name)])
        label_session.append(elmt.name)

Data = pd.DataFrame(l)
Data['year'] = year 
Data['gpName'] = gpNames
Data['gpNumber'] = gpNumbers
Data['sessionName'] = label_session

Data_driver = Data['Driver'].apply(pd.Series)
Data_driver_drop = Data_driver.drop(['url','givenName','familyName','dateOfBirth','permanentNumber'],axis=1)
Data_driver_drop =Data_driver_drop.rename(columns={"nationality": "DriverNationality"})

Data_constructor = Data['Constructor'].apply(pd.Series)
Data_constructor_drop = Data_constructor.drop(['url','name'],axis=1)
Data_constructor_drop = Data_constructor_drop.rename(columns={"nationality": "constructorNationality"})

Data_fast = Data['FastestLap'].apply(pd.Series)
Data_fast = pd.concat([Data_fast.drop(['AverageSpeed'], axis=1), Data_fast['AverageSpeed'].apply(pd.Series)], axis=1)
Data_fast = pd.concat([Data_fast.drop(['Time'], axis=1), Data_fast['Time'].apply(pd.Series)], axis=1)
Data_fast_drop = Data_fast.drop([0,'units'],axis=1)
Data_fast_drop = Data_fast_drop.rename(columns={"lap": "fastestLapNumber","rank":"fastestLapRank","speed":"fastestLapAvgSpeed","time":"fastestLapTime"})

Data_time = Data['Time'].apply(pd.Series)
Data_time_drop = Data_time.drop([0],axis=1)
Data_time_drop = Data_time_drop.rename(columns={"millis": "totalTime","time":"TimeInterval"})

df_global = pd.concat([Data.drop(['Driver'], axis=1), Data_driver_drop], axis=1)
df_global = pd.concat([df_global.drop(['Constructor'], axis=1), Data_constructor_drop], axis=1)
df_global = pd.concat([df_global.drop(['FastestLap'], axis=1), Data_fast_drop], axis=1)
df_global = pd.concat([df_global.drop(['Time'], axis=1), Data_time_drop], axis=1)

df_global["fastestLapTime"] = df_global['Q1'].fillna(df_global['fastestLapTime'])
df_global["fastestLapTime"] = df_global['Q2'].fillna(df_global['fastestLapTime'])
df_global["fastestLapTime"] = df_global['Q3'].fillna(df_global['fastestLapTime'])

#practices_df creation

lnames = df_global.columns
NaNs = [np.NaN for x in range(len(lnames))]

def GetPositionsDataframe(year, gpNumber, sessionLetters):
    
    session = None
    
    weekend = fastf1.get_session(year, gpNumber)
    if (sessionLetters[0] == "P"):
        session = weekend.get_practice(sessionLetters[1])
        
    laps = session.load_laps()
    drivers = pd.unique(laps['Driver'])
    
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = laps.pick_driver(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)
    
    df = pd.DataFrame(fastest_laps)
    df['Position'] = np.arange(1, df.shape[0] + 1)
    df['Weekend'] = session
    df['WeekendNumber'] = gpNumber
    return df

def DataFramePracticeResult(fastests):
    rows = []
    for i in range(fastests.shape[0]):
        row = dict(zip(lnames, NaNs))
        row["number"] = fastests.iloc[i]["DriverNumber"]
        row["position"] = fastests.iloc[i]["Position"]

        row["year"] = fastests.iloc[i]["Weekend"].weekend.year
        row["gpName"] = fastests.iloc[i]["Weekend"].weekend.name
        row["gpNumber"] = fastests.iloc[i]["WeekendNumber"]
        row["sessionName"] = fastests.iloc[i]["Weekend"].name

        #Le driverId et les autres infos sur l'écurie ne sont pas fournies par fastest laps

        row["code"] = fastests.iloc[i]["Driver"]

        #...

        row["fastestLapNumber"] = fastests.iloc[i]["LapNumber"]
        row["fastestLapRank"] = fastests.iloc[i]["Position"]
        #Fastest lap avg. speed needs a calculation
        row["fastestLapTime"] = fastests.iloc[i]["LapTime"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)

session_letter = ['P1', 'P2', 'P3']
firstrows = True
practices_df = None
for year in range(2018,2022):
    for gp in range(1,23):
        for l in session_letter:
            try:
                fastests = GetPositionsDataframe(year, gp, l)
                if (firstrows == True):
                    practices_df = DataFramePracticeResult(fastests)
                    firstrows = False
                else:
                    practices_df = pd.concat([practices_df, DataFramePracticeResult(fastests)])
                    
            except:
                print("problem on",year, gp, l)
                    
for year in df_global['year'].unique().tolist():
    for venue in df_global['gpName'].unique().tolist():
        for code in df_global['code'].unique().tolist():
            driverInfo = df_global[(df_global['code'] == code) & (df_global['year'] == year) & (df_global['gpName'] == venue)][['driverId', 'DriverNationality', 'constructorId','constructorNationality']]
            if (driverInfo.shape[0] > 0):
                driverInfoList = driverInfo.iloc[0].values.flatten().tolist()
                practices_df.loc[(practices_df['code'] == code) & (practices_df['year'] == year) & (practices_df['gpName'] == venue), ['driverId', 'DriverNationality', 'constructorId','constructorNationality']] = driverInfoList

#df_global and practices_df are merged into one dataframe

combinedDFS = pd.concat([df_global, practices_df])

combinedDFS = combinedDFS.sort_values(['year', 'gpNumber', 'sessionName'], ascending=[True, True, True])
combinedDFS = combinedDFS.reset_index(drop = True)

#.csv with all data available from 2018 to 2021
combinedDFS.to_csv(r'C:\Users\lmbfr\OneDrive\Documents\Travail ESILV\Cours + Exercices\Semestre 7\Python Data Analysis\F1\allData.csv', index=False)

#dataframe with only complete weekends (P1, P2, P3, Q, R)
sessionsRecordsNb = combinedDFS.groupby(['year', 'gpName', 'sessionName']).size()

dfSessionsRecordsNb = pd.DataFrame(sessionsRecordsNb)
dfSessionsRecordsNb.reset_index()

sessionsNumber = dfSessionsRecordsNb.groupby(['year', 'gpName']).size()
dfSessionsNumber = pd.DataFrame(sessionsNumber)
dfSessionsNumber.rename(columns={ dfSessionsNumber.columns[0]: "Number of Sessions" }, inplace = True)

dfCompleteWeekends = dfSessionsNumber[dfSessionsNumber['Number of Sessions'] == 5]
completeVenuesList = list(dfCompleteWeekends.to_dict()['Number of Sessions'].keys())

df_accurate = combinedDFS[combinedDFS[["year","gpName"]].apply(tuple, 1).isin(completeVenuesList)]
df_accurate = df_accurate.reset_index(drop = True)

df_accurate.to_csv(r'C:\Users\lmbfr\OneDrive\Documents\Travail ESILV\Cours + Exercices\Semestre 7\Python Data Analysis\F1\completeWeekendsData.csv', index=False)