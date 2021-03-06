# Disaster_Response_Pipeline

## 1. Introduction 
- Here disaster data from [Figure Eight](https://appen.com/) is analyzed to build a model for an API that classifies disaster messages. 
You can find a data set containing real messages that were sent during disaster events in the data folder. The aim is to create a machine learning pipeline to categorize these events so that the messages are recieved by an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


<p align="center">
<img src="figs/disaster-response-project1.png" width="75%" height="75%">
</p>
<p align="center">
<img src="figs/disaster-response-project2.png" width="75%" height="75%">
</p>


## 2. Project Components 

#### ETL Pipepline 
``` data/process_data.py```  writes a data cleaning pipeline that
- Loads the ```data/messages.csv``` and ```data/categories.csv``` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### ML Pipeline 
```python models/train_classifier.py ``` writes a machine learning pipeline that 
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### Flask Web App
Flask web app is provided in app folder as ```app/run.py``` with templates as ```app/templates/go.html app/templates/master.html```

## 3. Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database 
``` python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db ```

- To run ML pipeline that trains classifier and saves 
```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl``` 

2. Run the following command in the app's directory to run your web app. 
```python run.py```

3. Go to http://0.0.0.0:3001/

## 4.  Installations

- nltk 3.3.0
- numpy 1.15.2
- pandas 0.23.4
- scikit-learn 0.20.0
- sqlalchemy 1.2.12

## 5.  Acknowledgments

I am thankful for the challenge provided by Udacity.
