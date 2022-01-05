import fastf1
import fastf1.plotting
from fastf1.livetiming.data import LiveTimingData
from fastf1.core import Laps

import pandas as pd
import numpy as np
import math

from datetime import datetime, timedelta
from timple.timedelta import strftimedelta
import re
import mpld3

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.pyplot import figure
from matplotlib.ticker import FormatStrFormatter
import plotly.express as px
import plotly.graph_objects as go


df = None
races = None
driversCauses = None
teamList = None
teamColors = None
dictTeamColors = None
fTelemetry1 = None
fTelemetry2 = None
lapsDriver1 = None
lapsDriver2 = None

print(matplotlib.__version__, "matplotlib version // need to be higher than 3.4")

def init_viz():
    global df, races, driversCauses, teamList, teamColors, dictTeamColors, fTelemetry1, fTelemetry2, lapsDriver1, lapsDriver2

    fastf1.Cache.enable_cache(r'..\Data\Cache')

    #---> Data management
    df = pd.read_csv("../Data/allData.csv")

    races = df[df["sessionName"] == "Race"]

    driversCauses = ['Collision', 'Accident', 'Collision damage', 'Puncture', 'Disqualified', 'Withdrew', 'Spun off', 'Damage', 'Debris', 'Tyre', 'Out of fuel']

    #---> Team colors mapping for plots
    #teamColors = ['alfa romeo': '#900000', 'alphatauri': '#2b4562', 'alpine': '#0090ff', 'aston martin': '#006f62', 'ferrari': '#dc0000', 'force_india': '#F596C8', 'haas': '#ffffff', 'mclaren': '#ff8700', 'mercedes': '#00d2be', 'racing_point': '#F596C8', 'red bull': '#0600ef', 'renault': '#FFF500', 'sauber' : '#9B0000', 'toro_rosso': '#0032FF', 'williams': '#005aff']
    teamList = df['constructorId'].unique().tolist()
    teamList.sort()
    teamColors = ['#900000', '#2b4562', '#0090ff', '#006f62', '#dc0000', '#F596C8', '#fdfdfd', '#ff8700', '#00d2be', '#F596C8', '#0600ef', '#FFF500', '#9B0000', '#0032FF', '#005aff']
    dictTeamColors = dict(zip(teamList, teamColors))

    #Telemetry pre-load
    sessionComp = fastf1.get_session(2021, 1, 'Q')
    laps = sessionComp.load_laps(with_telemetry = True)
    
    lapsDriver1 = laps.pick_driver('NOR')
    fLap1 = lapsDriver1.pick_fastest()
    fTelemetry1 = fLap1.get_telemetry()
    
    lapsDriver2 = laps.pick_driver('SAI')
    fLap2 = lapsDriver2.pick_fastest()
    fTelemetry2 = fLap2.get_telemetry()

    #Race Laps pre-load
    sessionComp = fastf1.get_session(2021, 1, 'R')
    laps = sessionComp.load_laps()
    
    lapsDriver1 = laps.pick_driver('NOR')

    lapsDriver2 = laps.pick_driver('SAI')

    return df,races,driversCauses,dictTeamColors, fTelemetry1, fTelemetry2, lapsDriver1, lapsDriver2

#---> Session retrieval with regex help
#Checks first if number in [1,2,3] exists (--> Practice x), then searches for a "q" (--> Qualification)
#If didn't find any of these conditions (--> Race)
def RetrieveSession(year, gpNumber, sessionName):
    sessionFound = False
    
    digits = re.findall(r'\d', sessionName)
    if (len(digits) > 0):
        if (digits[0] in ['1', '2', '3']):
            sessionName = "Practice " + digits[0]
            sessionFound = True
        
    if (sessionFound == False):
        if (bool(re.search(r'q', sessionName, flags=re.IGNORECASE))):
            sessionName = "Qualifying"
        else:
            sessionName = "Race"
    
    sessionSelected = df[(df["year"] == year) & (df["gpNumber"] == gpNumber) & (df["sessionName"] == sessionName)]
    return sessionSelected

#---> Formatting methods
def StringToTimeDelta(time):
    try:
        if (time[0] == "+"):
            time = time[1:]
        digits = re.findall(r"\d+", time)
        if (len(digits) == 2):
            delta = timedelta(seconds = int(digits[0]), microseconds = int(digits[1]) * 1000)
        else:
            delta = timedelta(minutes = int(digits[0]), seconds = int(digits[1]), microseconds = int(digits[2]) * 1000)
        return delta
    except:
        return time

def FakeTimeDeltaToTimeDelta(time):
    try:
        min = time[10:12]
        sec = time[13:15]
        thousandth = time[16:19]
        delta = timedelta(minutes = int(min), seconds = int(sec), microseconds = int(thousandth) * 1000)
        return delta
    except:
        return time

def FirstListElement(l):
    if len(l) > 0:
        return int(l[0])
    else:
        return np.NaN

def RaceTimeInterval(x):
    seconds = int(re.findall(r'\d+', str(x))[0])
    if (seconds >= 60):
        seconds = f"{seconds//60}m{seconds%60}"
    return seconds

def TimeDeltaTotalSeconds(delta):
    try:
        return delta.seconds + delta.microseconds / 1000000
    except:
        return np.NaN


#---> Plots made with "races" dataframe
# Session needs to be retrieved before with RetrieveSession(year, gpNumber, sessionName)
# Number is only meant to define qualification session number
#plotly express version
def RankingDisplay(session, Number = None):
    
    fig = None

    title = ""
    
    team_colors = list()
    for index, row in session.iterrows():
        color = dictTeamColors[row["constructorId"]]
        team_colors.append(color)
        
    if (bool(re.search(r'practice', session["sessionName"].iloc[0], flags=re.IGNORECASE))):
        session['fastestLapTime'] = session['fastestLapTime'].apply(FakeTimeDeltaToTimeDelta)
        session['DeltaP'] = session['fastestLapTime'] - session['fastestLapTime'].min()
        
        fig = px.bar(session, x = 'DeltaP', y = 'code', orientation = 'h', color = 'code', color_discrete_sequence = team_colors)

        fastest = session[session["DeltaP"] == timedelta(0)]
        title = session["gpName"].iloc[0] + " " + str(session["year"].iloc[0]) + " " + str(session["sessionName"].iloc[0])
        title += "\nFastest Lap: " + strftimedelta(fastest["fastestLapTime"].iloc[0], '%m:%s.%ms') + " by " + fastest["driverId"].iloc[0].capitalize()
        
        
    
    elif (session["sessionName"].iloc[0] == "Qualifying"):
        session['Q1'] = session['Q1'].apply(StringToTimeDelta)
        session['Q2'] = session['Q2'].apply(StringToTimeDelta)
        session['Q3'] = session['Q3'].apply(StringToTimeDelta)

        session['DeltaQ1'] = session['Q1'] - session['Q1'].min()
        session['DeltaQ2'] = session['Q2'] - session['Q2'].min()
        session['DeltaQ3'] = session['Q3'] - session['Q3'].min()
        
        session = session[session[f'Q{Number}'].notnull()]
        
        fig = px.bar(session, x = f'DeltaQ{Number}', y = 'code', orientation = 'h', color = 'code', color_discrete_sequence = team_colors)

        fastest = session[session[f"DeltaQ{Number}"] == timedelta(0)]
        title = session["gpName"].iloc[0] + " " + str(session["year"].iloc[0]) + f" Qualifying {Number}"
        title += "\nFastest Lap: " + strftimedelta(fastest[f"Q{Number}"].iloc[0], '%m:%s.%ms') + " by " + fastest["driverId"].iloc[0].capitalize()
    
    else:
        finishTime = session["TimeInterval"].iloc[0]
        session["TimeInterval"].iloc[0] = '+0.0'
        session["TimeInterval"] = session["TimeInterval"].apply(StringToTimeDelta)

        session["LapsBack"] = session["status"].str.findall(r'\d').apply(FirstListElement)

        session["TimeBack"] = session["LapsBack"]*session["fastestLapTime"].apply(StringToTimeDelta).max()
        session["TimeInterval"] = session["TimeBack"].fillna(session["TimeInterval"])
        
        session.replace({pd.NaT: np.NaN}, inplace = True)
        
        sessionR = session[session["TimeInterval"].notnull()]
        sessionR["TimeInterval"] = pd.to_timedelta(sessionR["TimeInterval"])
        
        fig = px.bar(sessionR, x = 'TimeInterval', y = 'code', orientation = 'h', color = 'code', color_discrete_sequence = team_colors)
        
        title = sessionR["gpName"].iloc[0] + " " + str(sessionR["year"].iloc[0]) + " " + str(sessionR["sessionName"].iloc[0])
        title += "\nFastest : " + finishTime + " - " + sessionR["driverId"].iloc[0].capitalize()
        
    fig.update_layout(title = title)
    
    return fig
    

#Constructors is True for Constructors Standings, False otherwise
#plotly express version
def SeasonRankings(year, constructors):
    
    team_colors = list()
    team_colors2 = list()
    
    fig = None
    standings = None
    
    if (constructors):
        standings = pd.DataFrame(races[races["year"] == year].groupby('constructorId')['points'].sum().sort_values(ascending = False))

        for index, row in standings.iterrows():
                color = dictTeamColors[index]
                team_colors.append(color)
    else:
        driversPoints = pd.DataFrame(races[races["year"] == year].groupby(['code', 'constructorId'])['points'].sum().sort_values(ascending = False))
        driversPoints = driversPoints.reset_index()        
        duplicateDrivers = driversPoints[driversPoints.duplicated(['code'])]
        duplicateDrivers = duplicateDrivers.rename({'constructorId': 'constructorId2', 'points': 'points2'}, axis='columns')
        driversPoints = driversPoints[~driversPoints.isin(duplicateDrivers.copy())].dropna()
        standings = pd.merge(driversPoints.copy(), duplicateDrivers.copy(), on = 'code', how = 'left')
        standings["points2"].replace({np.NaN: 0}, inplace = True)
        standings["constructorId2"].replace({np.NaN: 'empty'}, inplace = True)
        standings['constructorId2'] = np.where(standings['constructorId2'] == 'empty', standings['constructorId'], standings['constructorId2'])
        standings['totalPoints'] = standings['points'] + standings['points2']
        standings = standings.sort_values(['totalPoints'], ascending = False)
        
        
        for index, row in standings.iterrows():
                color = dictTeamColors[row["constructorId"]]
                team_colors.append(color)
                
        for index, row in standings.iterrows():
                color = dictTeamColors[row["constructorId2"]]
                team_colors2.append(color) 
    
    title = f"Season {year}\n"
    if (constructors):
        title += f'Constructors Standings - Champions : {standings.index[0].capitalize()} with {standings["points"].iloc[0]} Points'
    else:
        title += f'Drivers Standings - Champion : {standings["code"].iloc[0]} with {standings["totalPoints"].iloc[0]} Points'
    
    total = None
    
    if (constructors):
        stat = 'points'
        fig = px.bar(standings, x = standings.index, y = stat, text = stat, title = title, color = standings.index, color_discrete_sequence = team_colors)
    else:
        stat = 'totalPoints'
        fig = px.bar(standings, x = 'code', y = stat, text = stat, title = title, color = 'code', color_discrete_sequence = team_colors)
    
    fig.update_traces(textposition='auto')
    
    return fig


#DNF is False if you want to display statistics without weekends when the race ended up with a DNF
#Plotly express version
def QualiRaceRelation(year, DNF):
    if (DNF):
        QRrelation = races[(races["year"] == year) & (races['positionText'] != 'R')].groupby('code')[['grid', 'position']].mean()
    else:
        QRrelation = races[races["year"] == year].groupby('code')[['grid', 'position']].mean()
    QRrelation = QRrelation.sort_values(['grid'])
    QRrelation["racecraftEdge"] = QRrelation["grid"] - QRrelation["position"]
    
    title = f"Grid position/Race result\n{year} F1 season"
    if (DNF):
        title += " excluding weekends where DNFs occurred for drivers"
    
    fig = go.Figure(data=[
        go.Bar(name='Grid position', x = QRrelation.index, y = QRrelation['grid']),
        go.Bar(name='Race result', x = QRrelation.index, y = QRrelation['position'])
    ])
    
    fig.update_layout(title = title, barmode='group', colorway = ['#1f77b4', '#ff7f0e'])
    
    return fig

#driversRelated is True if wanting to display DNFs related to the driver
#drivers is True if wanting to display the stat for drivers (individual overview)
#Plotly express version
def DNFCounter(year, driversRelated, drivers):
    
    team_colors = list()
    
    keyword = ''
    
    DNFs = races[(~races['status'].isin(['Finished', 'Illness'])) & (~races['status'].str.contains('\d'))]
    
    if (driversRelated):
        DNFs = DNFs[(DNFs["year"] == year) & (DNFs["status"].isin(driversCauses))]
    else:
        DNFs = DNFs[(DNFs["year"] == year) & (~DNFs["status"].isin(driversCauses))]
        
    if (drivers):
        DNFs = pd.DataFrame(DNFs.groupby(['code']).size()).rename({0: 'count'}, axis='columns')
        keyword = 'code'
    else:
        DNFs = pd.DataFrame(DNFs.groupby(['constructorId']).size()).rename({0: 'count'}, axis='columns')
        keyword = 'constructorId'
    
    driversList = races[(races["year"] == year)][keyword].unique().tolist()
    drv0list = list(zip(driversList, [0.01 for x in range(len(driversList))]))
    drv0 = pd.DataFrame(drv0list, columns = [keyword,'count'])
    drv0 = drv0.set_index(keyword)
    DNFs = pd.concat([DNFs, drv0])
    DNFs = DNFs[~DNFs.index.duplicated(keep='first')]
    
    DNFs = DNFs.sort_values(['count'])
    
    if (drivers):
        for index, row in DNFs.iterrows():
            if (row['count'] < 1):
                team_colors.append('lime')
            elif (row['count'] < 2):
                team_colors.append('greenyellow')
            elif (row['count'] < 3):
                team_colors.append('yellow')
            elif (row['count'] < 4):
                team_colors.append('orange')
            else:
                team_colors.append('red')
    else:
        for index, row in DNFs.iterrows():
            color = dictTeamColors[index]
            team_colors.append(color)
            
    title = f'{year} Season '
    if (drivers):
        title += "Drivers"
    else:
        title += "Constructors"
    title += " DNFs\nRetirements related to "
    if (driversRelated):
        title += "the driver"
    else:
        title += "car failures"
    
    fig = px.bar(DNFs, x = DNFs.index, y = 'count', title = title, text = 'count', color = DNFs.index, color_discrete_sequence = team_colors)
    fig.update_traces(textposition='auto')
    
    return fig



#Only displays teams results having stayed in F1 during the whole specified period
#Plotly express version
def ConstructorsForm(yearMin, yearMax):
    
    teamsPoints = races[(races["year"] >= yearMin) & (races["year"] <= yearMax)].groupby(['year', 'constructorId'])["points"].sum()
    teamsPercentages = teamsPoints.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    teamsPercentages = teamsPercentages.reset_index()
    teamsPercentages = teamsPercentages.round({'points': 2})
    teamsPercentages = teamsPercentages.rename({'points': 'pointsPortion'}, axis='columns')
    
    teamsPresence = pd.DataFrame(teamsPercentages['constructorId'].value_counts())
    teamsPresence = teamsPresence.rename({'constructorId': 'seasons'}, axis='columns')
    teamsCompleteData = teamsPresence[teamsPresence["seasons"] == (teamsPercentages['year'].max() - teamsPercentages['year'].min() + 1)].index
    
    teamsPercentages = teamsPercentages[teamsPercentages["constructorId"].isin(teamsCompleteData)].sort_values(['year', 'pointsPortion'], ascending = [True, False])
    
    title = f'Constructors Standings Points Percentage From {yearMin} to {yearMax}'
    
    fig = px.bar(teamsPercentages, x = 'year', y = 'pointsPortion', title = title, text = 'pointsPortion', color = 'constructorId', color_discrete_sequence = [dictTeamColors[team] for team in teamsPercentages['constructorId'].unique().tolist()])
    fig.update_traces(textposition='inside')
    
    return fig
        

#---> Telemetry plot
def DriversQualiComparison():
    
    sessionOverview = RetrieveSession(2021, 1, 'Q')
    teamColor1 = dictTeamColors[sessionOverview[sessionOverview['code'] == 'NOR']["constructorId"].iloc[0]]
    teamColor2 = dictTeamColors[sessionOverview[sessionOverview['code'] == 'SAI']["constructorId"].iloc[0]]
    
    #Haas(White) --> Light Grey to be visible
    if (teamColor1 == '#fdfdfd'):
        teamColor1 = 'grey'
    if (teamColor2 == '#fdfdfd'):
        teamColor2 = 'grey'
    
    #If comparing teammates with same team color, second driver = 'black'
    if (teamColor1 == teamColor2):
        teamColor2 = '#000000'
    
    fig, ax = plt.subplots(2,1,figsize = (15,20))
    ax[0].plot(fTelemetry1['Distance'], fTelemetry1['Speed'], color = teamColor1, label = 'NOR')
    ax[0].plot(fTelemetry2['Distance'], fTelemetry2['Speed'], color = teamColor2, label = 'SAI')
    ax[0].legend(loc = "best", fontsize = 12)
    title = f'  {sessionOverview["gpName"].iloc[0]} 2021 / {sessionOverview["sessionName"].iloc[0]}'
    plt.suptitle(title, fontsize = 14, y = 0.92)
    ax[0].set_title(f'Speed - Digital Comparison | NOR vs SAI', fontsize = 12)
    
    diff = fTelemetry1['Speed'] - fTelemetry2['Speed']
    trackColor = []

    for record in diff:
        if record > 0:
            trackColor.append(teamColor1)
        elif record < 0:
            trackColor.append(teamColor2)
        else:
            trackColor.append('lightgrey')
    for i in range(2,len(fTelemetry1)):
        ax[1].scatter(fTelemetry1.loc[i,'X'], fTelemetry1.loc[i,'Y'], color = trackColor[i])
    ax[1].set_title(f'Speed - Track Comparison | NOR vs SAI', fontsize = 12)

    plt.close()

    return mpld3.fig_to_html(fig)



#---> Laps plot
def RacePaceComparison():
    
    pace1 = lapsDriver1[['LapTime', 'LapNumber', 'PitInTime', 'Driver', 'Team']]
    pace1 = pace1.reset_index(drop = True)
    
    pace2 = lapsDriver2[['LapTime', 'LapNumber', 'PitInTime', 'Driver', 'Team']]
    pace2 = pace2.reset_index(drop = True)
    
    paceComp = pace1["LapTime"].apply(TimeDeltaTotalSeconds) - pace2["LapTime"].apply(TimeDeltaTotalSeconds)
    paceComp[0] = 0
    for i in range(len(paceComp)):
        if (np.isnan(paceComp[i])):
            lapbefore = paceComp[i - 1]
            if (i == len(paceComp) - 1):
                paceComp = lapbefore
            else:
                lapafter = paceComp[i + 1]
                if (abs(lapbefore) < abs(lapafter)):
                    paceComp[i] = lapbefore
                else:
                    paceComp[i] = lapafter
                
    s = RetrieveSession(2021, 1, 'R')
    
    teamColor1 = dictTeamColors[s[s["code"] == pace1["Driver"].iloc[0]]['constructorId'].iloc[0]]
    teamColor2 = dictTeamColors[s[s["code"] == pace2["Driver"].iloc[0]]['constructorId'].iloc[0]]
    
    #Haas(White) --> Light Grey to be visible
    if (teamColor1 == '#fdfdfd'):
        teamColor1 = 'grey'
    if (teamColor2 == '#fdfdfd'):
        teamColor2 = 'grey'
    
    #If comparing teammates with same team color, second driver = 'black'
    if (teamColor1 == teamColor2):
        teamColor2 = '#000000'
    
    fig = plt.figure(figsize=(20,10), dpi= 80)
    plt.fill_between(paceComp.index, paceComp, 0, where=paceComp >= 0, facecolor = teamColor1, interpolate=True, alpha=0.7, label = pace1["Driver"].iloc[0])
    plt.fill_between(paceComp.index, paceComp, 0, where=paceComp <= 0, facecolor = teamColor2, interpolate=True, alpha=0.7, label = pace2["Driver"].iloc[0])
    
    plt.ylim(-abs(max(paceComp)), abs(max(paceComp)))
    plt.xlabel("Race Lap Number")
    plt.ylabel("Delta in Seconds")
    plt.legend(loc = "best", fontsize = 12)
    
    title = f'{s["gpName"].iloc[0]} 2021 / {s["sessionName"].iloc[0]}\n'
    title += f'Pace Comparison | NOR vs SAI'
    plt.title(title, fontsize = 14)
    
    plt.close()

    return mpld3.fig_to_html(fig)

