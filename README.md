# F1_ML_Prediction
Formula one data analysis and prediction using machine learning techniques and FastF1 framework.  
Authors : Touitou Yves, Benjamin Porterie

<br/>

## Website opening procedure
Make sure you own all required packages/libraries to launch the project thanks to requirements.txt : "pip install -r requirements.txt"

Open the website with "python manage.py runserver" in your Python Terminal (you might need to change the terminal folder "cd F1_Prediction").

To display the website on internet, join the following link --> http://127.0.0.1:8000/

## Project Folder Organization
.csv files --> Data

Data Calculations and Displays --> F1_Prediction/Backend (/Viz.py for visualisation OR /predictions.py for prediction)

Django website --> F1_Prediction

Project's Powerpoint --> Powerpoint

FastF1 Notebooks --> Notebooks  

<br/>

### Project Outcome

At the start of the project, we had to deal with FastF1 data unstability.  
After a substantial formatting work, we obtained allData.csv (cf. "Practices_Records.ipynb" and "Dataset_Creation.ipynb").  
allData.csv gathers all sessions results from 2018 to 2021.


You can have an overview of any season or session from 2018 to 2021 thanks to "F1_Prediction/Backend/Viz.py", you can even compare two drivers performance on a particular venue.


You can predict any weekend's race from 2018 to 2021 with "F1_Prediction/Backend/predictions.py". After many models trials, we finally went for a Random Forest Regressor, with an Output Ranking transformation;  
Best parameters being : random state = 47, previous weekends considered = 17.

The most explicative variables to predict Sunday's race result are :
- Starting grid (easier to finish 1st when you start 1st)  
- FastestLapRankP2 (drivers make race pace trainings on harder tyre compounds used in Sunday's Race during "Practice 2" session)  
- FastestLapRankP3 (drivers make fastest lap and race pace trainings with softest tyre compound during "Practice 3" session)  
- FastestLapTimeP2 (same as "FastestLapRankP2" but the lap time is a bit less important than the rank)  
- Number (Number is driver's identity, shows racecraft importance in race finishing order)

Our prediction model has approximately a :
- 13.8% absolute accuracy
- 32.6% "-/+ 1 position" estimation accuracy
- 48.4% "-/+ 2 position" estimation accuracy
This is better than 5% which is the absolute random estimation mean accuracy.  
It's not necessarily a low percentage considering F1 is a motorsport, and various events can completely modify races finishing order.  
Our prediction model is more precise when predicting podium than the midfield or rear of the pack.  
Our prediction model is more precise as we make progress into the season, it can lean on previous races from the same season.
