# dsnd-DataEngineering
Repository for the Udacity Data Scientist Nanodegree's Data Engineering Disaster Response Pipeline Project

# Disaster Response Pipeline

### Installation:
Python nltk libraries:
 * punkt
 * wordnet
 * stopwords
 * WordNetLemmatizer
 * word_tokenize
 * PorterStemmer
 
Python sci-kit learn libraries:
* confusion_matrix
* GridSearchCV
* train_test_split
* RandomForestClassifier
* MultinomialNB
* Pipeline
* CountVectorizer
* TfidfTransformer
* classification_report
* MultiOutputClassifier

Python SQL libraries
* sqlite3
* sqlalchemy

General Python:
* re
* numpy
* pandas
* plotly

All nltk punkt, wordnet and stopwords are downloaded. Otherwise, the libraries are standard imports.

### File Descriptions:
./data:
  - process_data.py: Call with: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    1. Load messages.csv and categories.csv
    2. Clean / process the data into a SQL table resembling the format: 'ID',
        'Message', 'Original;, ' Genre', 'Cat1', ..., 'CatN'
    3. Store the table into a SQL DB with the provided filename
  - disaster_messages.csv:
    1. Message data in CSV format that contains: ID, Message, Original, Genre
  - disaster_categories.csv:
    1. Category classifiaction data that contains: ID, classification string for 36 categories
  - ETL Pipeline Preparation.ipynb:
    1. Jupyter Notebook of the `process_data.py` script. This file represents the initial work performed in order to 
       build the eventual python script.
  - DisasterResponse.db:
    1. SQL Database generated from the provided `messages.csv` and `categories.csv` data files. This will be overwritten whenever 
       `process_data.py` is run with the same name.
./models:
  - train_classifier.py: Call with: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    1. Load SQL database at the specified path
    2. Build, train, and predict on a model using the DB
    3. Evaluate the model
    4. Save the model as a pkl file in the specified location
  - classifier.pkl: Note - cannot be uploaded to github due to the size. Please run the `train_model.py` script.
    1. This is a pkl file for the model selected in `ML Pipeline Preparation.ipynb` and `train_classifier.py` that is based 
       on the best_estimator__ from the GridSearch and has the parameters noted below in Results.
  - ML Pipeline Preparation.ipynb:
    1. Jupyter Notebook of the `train_classifier.py` script. This file represents the intial work performed in order to build 
       the eventual python script. Additionally, it contains the work done with GridSearchCV to identify a more accurate (not 
       more efficient / optimum) model.
./app:
  - run.py:
    1. Extended template for a flask app that opens the `DisasterResponse.db` database, `classifier.pkl` model,
       and processes three plotly visualizations. Plots include Genre distribution counts, category
       distributions, and similar-themed category distribution skew.

### Results:
This project has three phases and therefore, three separate, but related sets of results.
1. The ETL Pipeline reads in message and category data, cleans it, and inserts it into a database. The
   current dataset results in 26,174 distinct messages and 35 categories. The 36th category (child_alone)
   has been dropped due to a complete lack of varying data (every message was classified as "not 'child_alon'".
2. The ML Pipeline reads in the database generated previosuly and prepares it for machine learning algorithms focused
   on text processing. Here, CountVectorizer and TfidfVectorizer are used with the custom tokenizer function (normalize,
   stem, lemmatize, remove stopwords) before being processes with RandomForrest or MultinomialNB classifier. Key parameters
   chosen include: Vect (max_df = 0.75), TFIDF (sublinear_tf = False) and CLF (n_estimators = 500, min_samples_lead = 1). 
   These were found using GridSearch to provide some level of accuracy improvements to the model at the expense of runtime.
   It was found that often, the Udacity workspace would timeout or go dormant waiting for the model to process completely. 
   Even when running the python scripts from a command line, the author's machine took several hours to complete.
   The following is based on the tuned model's classification matrices outputs:
   `
   `
3. The Webapp is a simple page that takes a message string and attempts to classify it. While accuracy is not bad (many
   categories had 85-95% accuracy when trained on 70% of the dataset, some issues were noticed when running new message strings.
   "My son is lost" and "My family is missing" were not able to be classified into the correct categories - "missing person"
   was never selected. This is likely due to the skew in the dataset which heavily leans towards categorization as "related" which
   is used for almost two thirds of the messages. Conversely, categories such as "missione people", "tools", "fire", or "hospitals"
   have under 1000, if not under 200, messages that were tagged to them. This means it is increasingly difficult for the model to 
   predict those messages correctly. Since messages can have multiple categories tied to them, the high number of "related" cases
   does not prevent other categories from being used, but it goes to demonstrate how little data there is for some of these categories.
   Prediction would need to be on data that was incredibly similar to the training set.
   
   The follwing plots are generated by the webapp. The first is a bar chart showing the distribution of categories that have been rolled
   up into four groups to demonstrate the general skew of the data.
   The second is the distribution of message category classifications which further shows how often some categories are used.
   The third is the default genre distribution plot provided by Udacity.
   ![newplot(2)](https://user-images.githubusercontent.com/49915194/123558040-3ae84700-d762-11eb-9819-bd0ee531fc4b.png)
   ![newplot(1)](https://user-images.githubusercontent.com/49915194/123558041-3ae84700-d762-11eb-8be3-7d18792cbfaf.png)
   ![newplot](https://user-images.githubusercontent.com/49915194/123558042-3ae84700-d762-11eb-8ea8-cb648de2ccc7.png)


### Instructions:
1. From the root directory of the project, run the following commands:
    - The follwing should be run first - it generates the database file used by the model. While the DB name can change,
      the Table inside should not.
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - The following can be run second to train the model and save it for use by the app. The classifier name should not be changed.
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To run the webapp, go to the app directory of the project and run the following. If the script hangs and the website does not load,
   try pressing CTRL-C to move it along. See Note in step 3.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/.
    - Note: it was found that the web app started very slowly, likely due to the model's size and that 'run_app.py' may hang. Crtl+C was
      used when the command line seemed to stop responding. A Single press helped it along, while multiple presses eventually killed the
      process. It is not known whether this is a bug on the author's machine or some other issue. The IDE provided by Udacity failed to 
      not timeout when attempting to run the app.

### Author:
Michael Partel

### Acknowledgements:
Data is provided by Figure Eight via Udacity. Much of the code is based on example code provided by the Udacity Data Science Nano Degree
class "Data Engineering".
