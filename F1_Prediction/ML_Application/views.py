from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import loader
#from Backend import Viz
from Backend import predictions as pred
from .forms import *
import logging

# Create your views here.


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

""" def Predictions_results(request,self):
    year = 2020
    grandPrix = request.GET.get('grandPrix','2')
    title= "Formula One Data visualization and prediction "
    years =[i for i in range(2018,2022)]
    year1 = years[0]
    year2 = years[-1]
    model = None
    if request.method == 'POST':

        form = YearForm(request.POST)
        form1 = EventForm(request.POST)

        if form.is_valid():

            year = form.cleaned_data['year']
            return HttpResponseRedirect(f'../Predictions/?year={year}')

        if form1.is_valid():
            grandPrix = form1.cleaned_data['grandPrix']


            return HttpResponseRedirect(f'/Predictions/?year={year}&?grandPrix={grandPrix}')
    grandPrix = request.GET.get('grandPrix','2')
    year = request.GET.get('year','2020')
    model_rd,y_pred,acc,X_test = pred.model_prevision_race(int(year),int(grandPrix))
    


    context = {
        'title' : title,
        'year': year,
        'year1' : year1,
        'year2' : year2,
        'form1': EventForm(),
        'form': YearForm(),
        'grandPrix': grandPrix,
        'y_pred': y_pred
    }
    return render(request, 'Predictions_results.html', context) """

def Predictions(request):
    year = request.GET.get('year','2020')
    grandPrix = 2
    title= "Formula One Data visualization and prediction"
    years =[i for i in range(2018,2022)]
    year1 = years[0]
    year2 = years[-1]
    if request.method == 'POST':

        form = YearForm(request.POST)
        form1 = EventForm(request.POST)

        if form.is_valid():

            year = form.cleaned_data['year']
            return HttpResponseRedirect(f'../Predictions/?year={year}')

        if form1.is_valid():
            grandPrix = form1.cleaned_data['grandPrix']
            return HttpResponseRedirect(f'/Predictions/?year={year}&?grandPrix={grandPrix}')

    grandPrix = request.GET.get('grandPrix','2')
    year = request.GET.get('year','2020')
    pred.init_from_local()
    model_rd,y_pred,acc,X_test = pred.model_prevision_race(int(year),int(grandPrix))

    context = {
        'title' : title,
        'year': year,
        'year1' : year1,
        'year2' : year2,
        'form1': EventForm(),
        'form': YearForm(),
        'grandPrix': grandPrix,
        'y_pred': y_pred
    }
    return render(request, 'Predictions.html', context)

def Visualization(request):
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