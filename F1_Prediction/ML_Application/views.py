from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import loader
#from Backend import Viz
from Backend import predictions
from .forms import *
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

def Predictions(request):
    titre = "Formula One Data visualization and prediction"
    years =[i for i in range(2018,2022)]
    year1 = years[0]
    year2 = years[-1]

    form = EventForm(request.POST)
    context = {
        'titre' : titre,
        'year1' : year1,
        'year2' : year2,
        'form': EventForm(),
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