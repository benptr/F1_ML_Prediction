from django import forms
from django.forms import Form


events = [
 (2018, 'Australian Grand Prix'),
 (2018, 'Bahrain Grand Prix'),
 (2018, 'Chinese Grand Prix'),
 (2018, 'Azerbaijan Grand Prix'),
 (2018, 'Spanish Grand Prix'),
 (2018, 'Monaco Grand Prix'),
 (2018, 'Canadian Grand Prix'),
 (2018, 'French Grand Prix'),
 (2018, 'Austrian Grand Prix'),
 (2018, 'British Grand Prix'),
 (2018, 'German Grand Prix'),
 (2018, 'Hungarian Grand Prix'),
 (2018, 'Belgian Grand Prix'),
 (2018, 'Italian Grand Prix'),
 (2018, 'Singapore Grand Prix'),
 (2018, 'Russian Grand Prix'),
 (2018, 'Japanese Grand Prix'),
 (2018, 'United States Grand Prix'),
 (2018, 'Mexican Grand Prix'),
 (2018, 'Brazilian Grand Prix'),
 (2018, 'Abu Dhabi Grand Prix'),
 (2019, 'Australian Grand Prix'),
 (2019, 'Bahrain Grand Prix'),
 (2019, 'Chinese Grand Prix'),
 (2019, 'Azerbaijan Grand Prix'),
 (2019, 'Spanish Grand Prix'),
 (2019, 'Monaco Grand Prix'),
 (2019, 'Canadian Grand Prix'),
 (2019, 'French Grand Prix'),
 (2019, 'Austrian Grand Prix'),
 (2019, 'British Grand Prix'),
 (2019, 'German Grand Prix'),
 (2019, 'Hungarian Grand Prix'),
 (2019, 'Belgian Grand Prix'),
 (2019, 'Italian Grand Prix'),
 (2019, 'Singapore Grand Prix'),
 (2019, 'Russian Grand Prix'),
 (2019, 'Japanese Grand Prix'),
 (2019, 'Mexican Grand Prix'),
 (2019, 'United States Grand Prix'),
 (2019, 'Brazilian Grand Prix'),
 (2019, 'Abu Dhabi Grand Prix'),
 (2020, 'Austrian Grand Prix'),
 (2020, 'Styrian Grand Prix'),
 (2020, 'Hungarian Grand Prix'),
 (2020, 'British Grand Prix'),
 (2020, '70th Anniversary Grand Prix'),
 (2020, 'Spanish Grand Prix'),
 (2020, 'Belgian Grand Prix'),
 (2020, 'Italian Grand Prix'),
 (2020, 'Tuscan Grand Prix'),
 (2020, 'Russian Grand Prix'),
 (2020, 'Eifel Grand Prix'),
 (2020, 'Portuguese Grand Prix'),
 (2020, 'Emilia Romagna Grand Prix'),
 (2020, 'Turkish Grand Prix'),
 (2020, 'Bahrain Grand Prix'),
 (2020, 'Sakhir Grand Prix'),
 (2020, 'Abu Dhabi Grand Prix'),
 (2021, 'Bahrain Grand Prix'),
 (2021, 'Emilia Romagna Grand Prix'),
 (2021, 'Portuguese Grand Prix'),
 (2021, 'Spanish Grand Prix'),
 (2021, 'Monaco Grand Prix'),
 (2021, 'Azerbaijan Grand Prix'),
 (2021, 'French Grand Prix'),
 (2021, 'Styrian Grand Prix'),
 (2021, 'Austrian Grand Prix'),
 (2021, 'British Grand Prix'),
 (2021, 'Hungarian Grand Prix'),
 (2021, 'Belgian Grand Prix'),
 (2021, 'Dutch Grand Prix'),
 (2021, 'Italian Grand Prix'),
 (2021, 'Russian Grand Prix'),
 (2021, 'Turkish Grand Prix'),
 (2021, 'United States Grand Prix'),
 (2021, 'Mexico City Grand Prix'),
 (2021, 'S??o Paulo Grand Prix'),
 (2021, 'Qatar Grand Prix'),
 (2021, 'Saudi Arabian Grand Prix'),
 (2021, 'Abu Dhabi Grand Prix')]

years = zip(set([x[0] for x in events]),set([x[0] for x in events]))
year1 = int
def update_year(year_ = 2020,*args, **kwargs):
    global year1
    year1= year_
    return year_

year1 = update_year()

def GPname_calculation(year):
        gpNames = [x[1] for x in events if x[0] == year]
        number = [x for x in range(1,1+22)]
        gpNumber = zip(number,number) 
        return gpNumber



class YearForm(Form):
    year = forms.ChoiceField(choices = years)
      
class EventForm(Form):
    
    grandPrix = forms.ChoiceField(choices = GPname_calculation(year1))
    


  
    

