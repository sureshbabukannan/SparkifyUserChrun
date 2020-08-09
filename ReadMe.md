# Sparkify Music portal - User Churn Modeling 
### For final project of Datascientist Nanodegree by [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025)

## Project Overview
Sparkify is a fictious online Music poral. The users can listen to songs and music as `Guest` or `Registered` users.

The `registered` user can be `free` or `paid` level users.  The users can **listen, add a friend, thumps up, thumbs down, add to favourite** to a song. 
User can also `Upgrade` to paid subscription and `Upgrade` multiple time to multiple levels of subscription, `Downgrade` to lower paid level subscription or `Downgrade` to free level 
. Users can also `Cancel` the subscription. All this activites are logged by the portal in `User Event log`.  Each event is logged as a log record. Also user can visit **home page, Save Settings, Help page**, can **login** and **logout**. 
These events are also logged by Sparkify Poral.

The users who are at paid level or free level can **Cancel** and leave the subscription. This call the **User Churn** or **attrition**. 

We are provided with 2 sample of the user log with `2` months data. **mini_sparkify_event_data.json** has `2 months` data of `229` users with size of `128 MB`.
**medium-sparkify-event-data.json** has `2 months` data with log of `499` users with size of `256MB`.
>>   **The Aim is this experiment is to predict the `user churn` by training Machine learning models with User log data.**

## Files in the Github Repo
> * [ReadMe.md](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/ReadMe.md) -- This file.
> * [Sparkify_IBMCloud_final.ipynb](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify_IBMCloud_final.ipynb) : **The main notebook of the project run in IBM cloud** and the analysis is based on **medium-sparkify-event-data.json** dataset. **Use this for any further work**
> * [Sparkify_IBMCloud_final.html](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify_IBMCloud_final.html) : The main notebook of the project saved as `HTML`. 
> * [Sparkify-Features-per-User.ipynb](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify-Features-per-User.ipynb) : Notebook with same features and approach as Main notebook but data is **mini-sparkify-event-data.json** dataset. Run on Udacity Workspace.
> * [Sparkify-Features-per-User.html](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify-Features-per-User.html) : Above Notebook saved as `HTML`.
> * [Sparkify-Features-per-User-WithStdScaling.ipynb](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify-Features-per-User-WithStdScaling.ipynb) : Notebook is based on mini-sparkify-event-data.json dataset and additioanally features are `Standard Scaled`. 
> * [Sparkify-Features-per-User-WithStdScaling.html](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify-Features-per-User-WithStdScaling.html) : The above notebook stored as `HTML`  
> * [Sparkify_Features_per_user_per_day.ipynb](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify_Features_per_user_per_day.ipynb) : This is the first attempt of the experiment where features are extracted `Per user` and further split into `per day` basis. It did produce the expected results so move on to having features as `Per user` basis only for later analysis.
> * [Sparkify_Features_per_user_per_day.html](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/Sparkify_Features_per_user_per_day.html) : Above Notebook saved as `HTML`.
> * [lib_version.txt](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/lib_version.txt) -- This is text file with the list of libraries in the enviroments used for the project and their versions 
> * [mini_sparkify_event_data.zip](https://github.com/sureshbabukannan/SparkifyUserChrun/blob/master/mini_sparkify_event_data.zip) -- Zip file of Data used for analysis  **mini_sparkify_event_data.json** ~10.6 MB

**medium-sparkify-event-data.json** used in the IBM cloud notebook is not provided here because of the huge size, You can download the data at [Udacity Workspace](https://classroom.udacity.com/nanodegrees/nd025-ent/parts/dc09e9a3-1794-4fb9-be8f-0b9082381c1f/modules/94cd233b-c676-4a6d-82d5-ce4f2db6d1a2/lessons/16dc9c16-b0a5-4d43-8def-ad24e34f156d/concepts/62031ed0-d82e-4c04-bf01-8430fa7957f8) after user authentication.
    
## Prerequesite Software 
- Anacoda 4.x or above / IBM watson studio / AWS EMR (Machine learning run time environment )
- Python 3.6.x (Programming language libraries)
- pySpark 2.4.x (Machine Learning library)
- matplotlib 3.03 (Data Visualisation library )
- seaborn-0.8.1 or later (graph ploting 
- pandas-0.23.3 (Dataframe library)
- numpy-1.12.1 (numerical calculations library)
- jupyter (Ipython notebook for Jupyiter notbook runtime)

- The full list of software in the enviroment can be found in `lib_version.txt` file in the Git Repo.

## Installation
> * For notebooks using **mini_sparkify_event_data.json** and Conda workspace
>>>> * If you like to create a seperate conda enviroment run the below command in Conda environment
```
conda create --name <your enviroment name> --file lib_version.txt
```

>>>> * Unzip the data file `**mini_sparkify_event_data.json** ` place it in the same folder as the `Sparkify-Features-per-User.ipynb` 

>>>> * Run the `jupyter notebook` command from a `Conda prompt` as shown

``` (base) Conda prompt> jupyter notebook Sparkify-Features-per-User.ipynb ```

>>>> * If a different dataset with the same schema is to be trained - Change the below line in the notebook to load different datasource and click run all under cell menu item

```
df = spark.read.json('<your_json_file_name_here.json>')
```

> * To use the **Sparkify_IBMCloud_final.ipynb** notebook
>>>> * create IBM watson studio project download and add **medium-sparkify-event-data.json** as asset to the Watson studio project
>>>> * Create a new notebook with code to access the medium-sparkify-event-data.json with in the notebook
>>>> * The add the contents of `Sparkify_IBMCloud_final.ipynb` to be new notebook created above and start executing.


## Problem Statement
**Evaluate algorithms and build a machine learning model to predict the users churn** using the user event log data. Choose a best model and predict user who could churn using the model.

## Metrics to Evaluate Model
The dataset taken for analyis is imbalanced dataset with only around `23%` cancelled user. As the dataset is slightly imbalanced 
to existing customers `F1-score` or `Harmonic Mean` is as used a primary metrics to evaluate the model.

## Strategy taken for Modelling 
The **mini-sparkify-event-data.json** is used in 3 different way to do EDA and build ML models. Then the approach producing the best result is
used on **medium-sparkify-event-data.json** and run in `IBM Watson studio` to produce the final output in `Sparkify_IBMCloud_final.ipynb`

#### Strategy #1 - On Udacity Workspace
> * Using **mini-sparkify-event-data.json** do EDA and features extracted as `per user per day` basis.  
> * There are  `users x number days per user` number of records in features dataset for ML model input.
> * The output is `Sparkify_Features_per_user_per_day.ipynb`  
> * This approach produced as maximum `F1` score of `0.29`

#### Strategy #2 - On Udacity Workspace
> * Using **mini-sparkify-event-data.json** do EDA and features extracted as `per user` basis.  
> * The number of records in features dataset for ML model input is same as number of users , that is 229.
> * `No Scaling` applied to the features, as the feature are in comparable scale like time in seconds, days and event counts
> * The output is `Sparkify-Features-per-User.ipynb`  
> * This approach produced as maximum `F1` score of `0.51`
 
#### Strategy #3 - On Udacity Workspace
> * Using **mini-sparkify-event-data.json** do EDA and features extracted as `per user` basis.  
> * The number of records in features dataset for ML model input is same as number of users , that is 229.
> * Standard Scaling is applied to the features to verify if there are any difference with results without scaling 
> * The output is `Sparkify-Features-per-User.ipynb`  
> * This approach produced as maximum `F1` score of `0.49`

Out of above 3 approaches `Strategy #2` produced most favourable result. 
This approach is used on **medium-sparkify-event-data.json** producing `Sparkify_IBMCloud_final.ipynb`.

## Data Analysis
These are the fields of the **user log** provide as json file **mini-sparkify-event-data.json**

> The user log has schema as follows.
```
root
|-- artist: string (nullable = true)
|-- auth: string (nullable = true)
|-- firstName: string (nullable = true)
|-- gender: string (nullable = true)
|-- itemInSession: long (nullable = true)
|-- lastName: string (nullable = true)
|-- length: double (nullable = true)
|-- level: string (nullable = true)
|-- location: string (nullable = true)
|-- method: string (nullable = true)
|-- page: string (nullable = true)
|-- registration: long (nullable = true)
|-- sessionId: long (nullable = true)
|-- song: string (nullable = true)
|-- status: long (nullable = true)
|-- ts: long (nullable = true)
|-- userAgent: string (nullable = true)
|-- userId: string (nullable = true)
```
At the start of the anlysis `the` **day** `of the year` is add as additional field for data analysis
```
|-- day: integer (nullable = true)
```

The dataset is checked for `nulls` in `UserId` and `page` datapoints and event log with null values for these 2 datafields are removed.

**EDA** Intial Exploratarory Data Analyis shows that  **mini-sparkify-event-data.json** dataset has `229` users and `2 months` of event log in the dataset. The number of free and paid 
registered user and free and paid cancelled users are obsevered.

## Data Visualisation
The log dataset is grouped by `per user per user` basis to find the following weekly metrics. 
Visulatisation are created based on these metrics.
    
> * `Per Cancelled user Weekly song listening time`
> * `Per Active user Weekly song listening time`
> * `Propotional Weekly number of Sessions per Cancelled user`
> * `Propotional Weekly number of Sessions per active user`
> * `Propotional Weekly Items navigated per Cancelled user`
> * `Propotional Weekly Items navigated per Active user`
> * `Weekly Thumbs Up per Cancelled user`
> * `Weekly Thumbs Up per Active user`
> * `Weekly Thumbs Down per Cancelled user`
> * `Weekly Thumbs Down per Active user`
> * `Weekly Add Friends per Cancelled user`
> * `Weekly Add Friends per Active user`
> * `Weekly number of Advertisement per Cancelled user`
> * `Weekly number of Advertisement per Active user`
> * `Total Cancelled user by Song`
> * `Total Downgraded user by Song`
> * `Total Upgraded user by Song`

Then extract the OS and Browser values for `User Agent` column and add as column to user log dataset.  Combine user and browser into 1 files to get 
combination of OS-Browser as 1 field

Based on OS and Browser files the following analysis is done.
> * `Propotional Number of cancellation by per user of OS`
> * `Propotional Number of cancellation by per user of Browser`
> * `Propotional Number of cancellation by per user with OS-Browser combination`

The Visualisation are part of the workbook, gives a fair idea on `how the Cancelled and Active users actitives varied over time`.
 
## Features extraction

#### * For `per user per user` features for stategy #1 above producing `Sparkify_Features_per_user_per_day.ipynb`
Based of the EDA I have decideded to further granualarise the time bucket to a day per user. This will captures users behaviour more prescisely
over the period of the log dataset used, making full use of Time Seriese dataset. The same metricts explored by week in EDA is to be use by as features 
machine learing model but now the `varition of metrics is observed at each day interval`.

> **Numerical Features**
>> * Daily listening time in minitues - representing Time spent in listening to songs
>> * Number of sessions daily - How often user logoff and logged in every day
>> * Number of items in session daily - How much user was navigating daily
>> * Number of songs daily - Number of song user listen daily, this is different for how long because use could just play few minutes of the song and skip to nextsong
>> * Number of thumbs up daily
>> * Number of thumbs downs daily
>> * Number of upgrades daily
>> * Number of roll adverts daily
>> * Number of downgrades daily
>> * Number of add friends daily
>> * Number of add to playlist daily
>> * Number of distinct artist daily
>> * Number of distinct songs daily
>> * Sum of Ranks of song listened for the day by cancelled user - this will be zero for active users
>> * Sum of Ranks of all the song listened the day downgrade user - this will be zero fpr active user
>> * Length of time between upgrade and cancellation - This will be 0 for user who has not cancelled
>> * Lenght of time between downgrades - this will be 0 for user who are free or user how have not downgraded
>> * Lenght of time between last downgrade and cancellation - this will be 0 if not downgraded and not cancelled cancelled users

> **Categorical Feature extraction**
>> Numerical encoded of `OSBrowser` column. There are 7 disitnct values of `OSBrowser` and 7 feature columns.

At the end of the feature engineering we will have `1` feature vector `per user per day` of data in the dataset.
The classification label is `isCancelled` columns with values `0` for Active user `1` for cancelled user.

Features dataframe is saved as `features.csv` for use in later use.

#### * For `per user` features for stategy #2 & #3 above producing `Sparkify_Features_per_user.ipynb`

The log dataset is grouped by `per user` basis to extract the features. The catergorical features are same as for stategy#1

At the end of the feature engineering we will have `1` feature vector `per user` of data in the dataset. that is `225` features vectors.
The feature values of grouped at `user` level instead of `user and day` level.

The classification label is `isCancelled` columns with values `0` for Active user `1` for cancelled user.


## Modeling 

The following binary Classification alogrithms are used

>> * **Logisctic regression**
>> * **Random Forests**
>> * **Gradient-Boosted Trees**
>> * **Decision Tree Classifier**

> * The features dataset is loaded from the saved file. Loading back from a `.csv` format file resets the feature datatypes to `string`.
The `string` datatype is changed to `float` Type for all feature columns.

> * Features Vector is created using `VectorAssembler`. Some of the vectors are produced as `sparse vector` as output of `VectorAssembler`
 The `sparseVector` is converted in to `DenseVector` to be used in machine learning training.

> Detais of 3 strategies adopted for modelling are as follows 

#### **Stategy #1** producing `Sparkify_Features_per_user_per_day.ipynb`
>> Features dataframe is `per user per user` with around 3200 records.
>> `features` dataframe is split by `70 / 30` proporation into `test` and `train` datasets.
>> 4 Models are trained using `CrossValidator` with a `3 fold interation` and evaluator as `BinaryClassificationEvaluator`
```
     Evaluated_model = CrossValidator(estimator=model_in,
                                     estimatorParamMaps=param_grid,
                                     evaluator=BinaryClassificationEvaluator(),
                                     numFolds=3,
                                     seed=77
                                    )
```
>> * The `Crossvalidator` picks the best comination of `hyperparameter` of the model and whole of `train` dataset is used to train the models. 
>> * Prediction are made with `test` dataset `feature` and prediction is compared with `actual` value of `label` column.
Based on the `prediction` outcome and the `actual` value of the `label` column of `test` dataset following `metrics` are calculated for each model
>>> * **Accuracy**
>>> * **Precision**
>>> * **Recall**
>>> * **F1-Score**                 

>> * The model metrics are plotted for comparision using `seaborn` and `F1` score is used as `model selection criteria`. The model with the 
largest `F1` score is selected.

In this approach best model is `Gradinent Boosted Trees` and `Random forest algorthim` with F1 score of `0.32` on Validation set.

#### **Stategy #2** producing `Sparkify_Features_per_user.ipynb`

>> Features dataframe is `per user` with `226` records.
>> `features` dataframe is split by `70 / 30` proporation into `test` and `train` datasets.
>> 4 Models are trained using `CrossValidator` with a `3 fold interation` and evaluator as `BinaryClassificationEvaluator`
```
     Evaluated_model = CrossValidator(estimator=model_in,
                                     estimatorParamMaps=param_grid,
                                     evaluator=BinaryClassificationEvaluator(),
                                     numFolds=3,
                                     seed=77
                                    )
```
>> * The `Crossvalidator` picks the best comination of `hyperparameter` of the model and whole of `train` dataset is used to train the models. 
>> * Prediction are made with `test` dataset `feature` and prediction is compared with `actual` value of `label` column.
Based on the `prediction` outcome and the `actual` value of the `label` column of `test` dataset following `metrics` are calculated for each model
>>> * **Accuracy**
>>> * **Precision**
>>> * **Recall**
>>> * **F1-Score**                 

>> * The model metrics are plotted for comparision using `seaborn` and `F1` score is used as `model selection criteria`. The model with the 
largest `F1` score is selected.

In this approach best model is `Gradinent Boosted Trees` with F1 score of `0.54` on Validation set.

#### **Stategy #3** producing `Sparkify-Features-per-User-WithStdScaling.ipynb`

>> Features dataframe is `per user` with `226` records.
>> Features are scaled using `StandardScaler`
>> `features` dataframe is split by `70 / 30` proporation into `test` and `train` datasets.
>> 4 Models are trained using `CrossValidator` with a `3 fold interation` and evaluator as `BinaryClassificationEvaluator`
```
     Evaluated_model = CrossValidator(estimator=model_in,
                                     estimatorParamMaps=param_grid,
                                     evaluator=BinaryClassificationEvaluator(),
                                     numFolds=3,
                                     seed=77
                                    )
```
>> * The `Crossvalidator` picks the best comination of `hyperparameter` of the model and whole of `train` dataset is used to train the models. 
>> * Prediction are made with `test` dataset `feature` and prediction is compared with `actual` value of `label` column.
Based on the `prediction` outcome and the `actual` value of the `label` column of `test` dataset following `metrics` are calculated for each model
>>> * **Accuracy**
>>> * **Precision**
>>> * **Recall**
>>> * **F1-Score**                 

>> * The model metrics are plotted for comparision using `seaborn` and `F1` score is used as `model selection criteria`. The model with the 
largest `F1` score is selected.

In this approach best model is `Decision Trees Classsifer` with F1 score of `0.51` on Validation set.

##### Based on the 3 different stretegies `Stategy #3` is choose for run with IBM watson studio training with `medium-sparkify-event-data.json` 

## Conclusion
The experiment is done with a small set of data of `2` months. The features were computed per user basis. 
There are `29` features extracted from raw log data. The featured dataframe is then used to for ML modelling. 
There are `4` binary classification algorthms used to evaluate the metrics. Out of 4 models **Gradient Boosted tree and Decision Trees Classifier**
proved to be better in all metrics, with F1 score of `0.40` on Validation data and `0.43` on training data. However, There is scope to do further 
hyperparameter tuning and also fit the model with larger dataset.

Initial, I set out to work with mini-sparkify-event-data.json to understand data with EDA and algorithm suitable for this data. I used 3 stagergies to model the data.

    I used same set of features but with per day per user basis and trained the features with this model.
    I used the same set of features but with per user basis without splitting them on per day basis with standard scaling of features.
    I used the same set of features but with per user basis without splitting them on per day basis but without the standard scaling of features Strategy#3 gave a f1 score of 0.51 with mini-sparkify-event-data.json

    I uses the Strategy#3 but with medium-sparkify-event-data.json dataset on IBM cloud which gave F1 score of 0.40.
    It is recommended to use Gradient Boosted tree and Decision Trees Classifier for further hyperparameter tuning and fitting with full dataset for better F1 score.

Further, The model also needs to be pacakged as class and pacakges for deployment.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
**Copyright (c) 2020** Sureshbabu K

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Credits
* Udacity Mentors and Udacity Help forums for guidence.

* I have heavily referenced online documentation of pySpark, Pandas, Stackoverflow, Databrick and other techincal content online

* The license text choosen from [choosealicense](https://choosealicense.com/licenses/mit/)


**For further information on the Data Analysis and Machine learing modeling outtcome please refer to [The Sparkify blog](https://sureshbabukannan.github.io/SparkifyBlog/).**

