from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('Predictions/',views.Predictions,name= 'Predictions'),
   # path('Predictions/<str:year>',views.Predictions,name= 'Predictions'),
  # path('Predictions/<str:year>&<str:grandPrix>',views.Predictions_results,name= 'Predictions_results'),
    path('Visualization/',views.Visualization, name= 'Visualization'),
    path('ModelSpe/',views.ModelSpe, name= 'ModelSpe')
]