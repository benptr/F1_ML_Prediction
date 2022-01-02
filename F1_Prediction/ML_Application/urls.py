from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('Predictions/',views.Predictions,name= 'Predictions'),
    path('Visualization/',views.Visualization, name= 'Visualization'),
    
]