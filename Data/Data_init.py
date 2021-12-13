import fastf1
import pandas as pd

fastf1.Cache.enable_cache(r'C:\Users\benja\OneDrive - De Vinci\S7_ESILV\cach_directory')

session_letter = ['FP1', 'FP2', 'FP3', 'Q', 'R']
sessions = []
for year in range(2014,2021):
    for gp in range(1,22):
        for l in session_letter:
            try:
                session = fastf1.get_session(year, gp, l)
                sessions.append(session)
            except:
                print("problem on",year, gp, l)

l=[]
year = []
gp = []
label_session = []

for elmt in sessions:
    for res in elmt.results:
        l.append(res)
        year.append(elmt.weekend.year)
        gp.append(elmt.weekend.name)
        label_session.append(elmt.name)

Data = pd.DataFrame(l)
Data['year'] = year 
Data['gpName'] = gp
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

try : 
    df_global.to_csv(r'C:\Users\benja\OneDrive - De Vinci\S7_ESILV\Python\ProjetF1\F1_ML_Prediction\Data\data.csv')
    raise Exception
except Exception:
    print("a problem as occured ",Exception)

