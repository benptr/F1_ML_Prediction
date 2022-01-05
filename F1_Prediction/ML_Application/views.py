from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import loader
from Backend import Viz as viz
import mpld3
import plotly.express as px
import plotly.graph_objects as go
from Backend import predictions as pred
from .forms import *
import logging
import pandas as pd 
import json

# Create your views here.
year = 2020
name = ''
def index(request):
    titre = "Formula One Data visualization and prediction"
    years =[i for i in range(2018,2022)]
    year1 = years[0]
    year2 = years[-1]

    context = {
        'titre' : titre,
        'year1' : year1,
        'year2' : year2,
    }

    return render(request, 'index.html', context)
def ModelSpe(request):
    titre = "Formula One Data visualization and prediction"
    years =[i for i in range(2018,2022)]
    year1 = years[0]
    year2 = years[-1]
    pred.init_from_local()
    graph =  pred.test_deepness_Unique(17)
    context = {
        'titre' : titre,
        'year1' : year1,
        'year2' : year2,
        'graph' : graph,
    }

    return render(request, 'ModelSpe.html', context)

def Predictions(request):
    grandPrix = request.GET.get('grandPrix','3')
    year = request.GET.get('year','2020')

    title= "Formula One Data visualization and prediction"
    years =[i for i in range(2018,2022)]
    year1 = years[0]
    year2 = years[-1]
    if request.method == 'POST':

        form = YearForm(request.POST)
        form1 = EventForm(request.POST)
        

        if form.is_valid():
            
            year = form.cleaned_data['year']
            return HttpResponseRedirect(f'../Predictions/?year={year}&grandPrix={grandPrix}')

        if form1.is_valid():
            grandPrix = form1.cleaned_data['grandPrix']
            return HttpResponseRedirect(f'/Predictions/?year={year}&grandPrix={grandPrix}')




    pred.init_from_local()
    
    name = ''
    model_rd,y_pred,acc,X_test,X_train,param,results,results_json  = [],[],[],[],[],[],[],[]
    try:
        
        model_rd,y_pred,acc,X_test,X_train = pred.model_prevision_raceV2(int(year),int(grandPrix),17)
        
        results = pd.DataFrame(columns=['Position','Prediction'])
        results['Prediction'] =y_pred
        if len(y_pred) == 20 :
            results['Position'] =[i for i in range(1,21)]
            results['Correct'] = ['yes' if x == y_pred[x-1] else 'no' for x in [i for i in range(1,21)]]
            results['Correct_Interval_1'] = ['yes' if ((x == y_pred[x-1]) |((x+1) == y_pred[x-1])|((x-1) == y_pred[x-1])) else 'no' for x in [i for i in range(1,21)]]
            results['Correct_Interval_2'] = ['yes' if (((x-2) == y_pred[x-1]) |((x+2) == y_pred[x-1])|(x == y_pred[x-1]) |((x+1) == y_pred[x-1])|((x-1) == y_pred[x-1])) else 'no' for x in [i for i in range(1,21)]]
        else:
            results['Position'] =[i for i in range(1,20)]
            results['Correct'] = ['yes' if x == y_pred[x-1] else 'no' for x in [i for i in range(1,20)]]
            results['Correct_Interval_1'] = ['yes' if ((x == y_pred[x-1]) |((x+1) == y_pred[x-1])|((x-1) == y_pred[x-1])) else 'no' for x in [i for i in range(1,20)]]
            results['Correct_Interval_2'] = ['yes' if (((x-2) == y_pred[x-1]) |((x+2) == y_pred[x-1])|(x == y_pred[x-1]) |((x+1) == y_pred[x-1])|((x-1) == y_pred[x-1])) else 'no' for x in [i for i in range(1,20)]]
        results['Driver'] = pred.data_driver(X_test)
        #results = results.to_html()
        json_records = results.reset_index().to_json(orient ='records')

        results_json = json.loads(json_records)
        name = pred.get_name(X_test)
        X_test = X_test.to_html()
        X_train = X_train.to_html()
        param = model_rd.get_params()
    except:
        model_rd,y_pred,acc,X_test,X_train,param = ['Grand prix number not valid'],['Grand prix number not valid'],['Grand prix number not valid'],['Grand prix number not valid'],['Grand prix number not valid'],['']
    context = {
        'title' : title,
        'year': year,
        'year1' : year1,
        'year2' : year2,
        'form1': EventForm(),
        'form': YearForm(),
        'grandPrix': grandPrix,
        'y_pred': y_pred,
        'model_rd': model_rd,
        'model_params': param,
        'X_test' : X_test,
        'y_test' : [i for i in range(1,21)],
        'results': results_json,
        'name':name,
        'X_train':X_train


    }
    return render(request, 'Predictions.html', context)

def Visualization(request):
    titre = "Formula One Data visualization and prediction"
    years = [i for i in range(2018,2022)]
    year1 = years[0]
    year2 = years[-1]
    yearDefined = 2021
    gpNumberDefined = 1
    driver1Defined = 'NOR'
    driver2Defined = 'SAI'

    viz.init_viz()

    graph = [
        viz.RankingDisplay(viz.RetrieveSession(yearDefined, gpNumberDefined, 'P1')).to_html(full_html = False),
        viz.RankingDisplay(viz.RetrieveSession(yearDefined, gpNumberDefined, 'P2')).to_html(full_html = False),
        viz.RankingDisplay(viz.RetrieveSession(yearDefined, gpNumberDefined, 'P3')).to_html(full_html = False),
        viz.RankingDisplay(viz.RetrieveSession(yearDefined, gpNumberDefined, 'Q'), 1).to_html(full_html = False),
        viz.RankingDisplay(viz.RetrieveSession(yearDefined, gpNumberDefined, 'Q'), 2).to_html(full_html = False),
        viz.RankingDisplay(viz.RetrieveSession(yearDefined, gpNumberDefined, 'Q'), 3).to_html(full_html = False),
        viz.RankingDisplay(viz.RetrieveSession(yearDefined, gpNumberDefined, 'R')).to_html(full_html = False),

        viz.SeasonRankings(yearDefined, True).to_html(full_html = False),
        viz.SeasonRankings(yearDefined, False).to_html(full_html = False),

        viz.QualiRaceRelation(yearDefined, True).to_html(full_html = False),
        viz.QualiRaceRelation(yearDefined, False).to_html(full_html = False),

        viz.DNFCounter(yearDefined, False, False).to_html(full_html = False),
        viz.DNFCounter(yearDefined, True, False).to_html(full_html = False),

        viz.ConstructorsForm(year1, year2).to_html(full_html = False),

        viz.DriversQualiComparison(),

        viz.RacePaceComparison()
    ]

    context = {
        'titre' : titre,
        'year1' : year1,
        'year2' : year2,
        'yearDefined' : yearDefined,
        'gpNumberDefined' : gpNumberDefined,
        'driver1Defined' : driver1Defined,
        'driver2Defined' : driver2Defined,
        'graph' : graph
    }

    return render(request, 'Visualization.html', context)