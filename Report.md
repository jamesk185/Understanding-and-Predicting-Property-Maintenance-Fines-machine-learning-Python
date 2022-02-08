
# Understanding and Predicting Property Maintenance Fines

By James Kowalik

Completed as assignment 4 in the *Applied Machine Learning* course by University of Michigan

### Project Outline

The following project description is as is stated on the course assignment page.

This assignment is based on a data challenge from the Michigan Data Science Team (MDST).

The Michigan Data Science Team (MDST) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences (MSSISS) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. Blight violations are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?

The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.

Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.

The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).

Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.

For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using readonly/train.csv. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from readonly/test.csv will be paid, and the index being the ticket_id.

### Data fields

##### train.csv & test.csv

ticket_id - unique identifier for tickets <br>
agency_name - Agency that issued the ticket <br>
inspector_name - Name of inspector that issued the ticket <br>
violator_name - Name of the person/organization that the ticket was issued to <br>
violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred <br>
mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator <br>
ticket_issued_date - Date and time the ticket was issued <br>
hearing_date - Date and time the violator's hearing was scheduled<br> 
violation_code, violation_description - Type of violation<br>
disposition - Judgment and judgement type<br>
fine_amount - Violation fine amount, excluding fees<br>
admin_fee - \$20 fee assigned to responsible judgments <br>
state_fee - \$10 fee assigned to responsible judgments<br>
late_fee - 10% fee assigned to responsible judgments <br>
discount_amount - discount applied, if any <br>
clean_up_cost - DPW clean-up or graffiti removal cost<br>
judgment_amount - Sum of all fines and fees<br>
grafitti_status - Flag for graffiti violations<br>

##### train.csv only

payment_amount - Amount paid, if any <br>
payment_date - Date payment was made, if it was received <br>
payment_status - Current payment status as of Feb 1 2017 <br>
balance_due - Fines and fees still owed<br>
collection_status - Flag for payments in collections <br>
compliance [target variable for prediction]  <br>
&emsp; Null = Not responsible<br>
&emsp; 0 = Responsible, non-compliant<br>
&emsp; 1 = Responsible, compliant<br>
compliance_detail - More information on why each ticket was marked compliant or non-compliant<br>

### Data Processing

First I load in the data, remove variables that are unavailable in the test set and examine null value occurrences.


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

training_data = pd.read_csv('readonly/train.csv', encoding='latin1')
testing_data = pd.read_csv('readonly/test.csv')
address_data = pd.read_csv('readonly/addresses.csv')
latlon_data = pd.read_csv('readonly/latlons.csv')

unavailable_vars = ['payment_amount', 'payment_date', 'payment_status', 
                   'balance_due', 'collection_status', 'compliance_detail']
training_data.drop(unavailable_vars, inplace=True, axis=1)

training_data.isnull().sum()/len(training_data)
```




    ticket_id                     0.000000
    agency_name                   0.000000
    inspector_name                0.000000
    violator_name                 0.000136
    violation_street_number       0.000000
    violation_street_name         0.000000
    violation_zip_code            1.000000
    mailing_address_str_number    0.014390
    mailing_address_str_name      0.000016
    city                          0.000000
    state                         0.000372
    zip_code                      0.000004
    non_us_str_code               0.999988
    country                       0.000000
    ticket_issued_date            0.000000
    hearing_date                  0.049903
    violation_code                0.000000
    violation_description         0.000000
    disposition                   0.000000
    fine_amount                   0.000004
    admin_fee                     0.000000
    state_fee                     0.000000
    late_fee                      0.000000
    discount_amount               0.000000
    clean_up_cost                 0.000000
    judgment_amount               0.000000
    grafitti_status               0.999996
    compliance                    0.361262
    dtype: float64



So from here I can remove some null-dominant variables.


```python
training_data.drop(['violation_zip_code', 'grafitti_status', 'non_us_str_code'], 
                   inplace=True, axis=1)
testing_data.drop(['violation_zip_code', 'grafitti_status', 'non_us_str_code'], 
                   inplace=True, axis=1)
```

Removing rows where compliance is null.


```python
training_data = training_data.loc[~training_data['compliance'].isnull()]
```

Assuming all data is from Detroit and thus removing unnecessary address variables while noting that some relevant address info may be captured in lat/lon data.


```python
training_data.drop(['city', 'state', 'zip_code', 'country'],
                   inplace=True, axis=1)
testing_data.drop(['city', 'state', 'zip_code', 'country'],
                   inplace=True, axis=1)
```

Adding lat and lon variables by merging while removing address variable.


```python
addlanlon_data = address_data.merge(latlon_data, how='inner', on='address')
addlanlon_data.drop('address', inplace=True, axis=1)
training_data = training_data.merge(addlanlon_data, how='inner', on='ticket_id')
testing_data = testing_data.merge(addlanlon_data, how='inner', on='ticket_id')
```

Creating new binomial variable that indicates whether the mailing address and violation address street name are the same and then removing unnecessary variables.


```python
training_data['violadd_equal_mailadd'] = training_data['violation_street_name']==training_data['mailing_address_str_name']
training_data['violadd_equal_mailadd'] = training_data['violadd_equal_mailadd'].astype('uint8')
training_data.drop(['violation_street_number', 'violation_street_name',
                   'mailing_address_str_number', 'mailing_address_str_name'],
                  inplace=True, axis=1)

testing_data['violadd_equal_mailadd'] = testing_data['violation_street_name']==testing_data['mailing_address_str_name']
testing_data['violadd_equal_mailadd'] = testing_data['violadd_equal_mailadd'].astype('uint8')
testing_data.drop(['violation_street_number', 'violation_street_name',
                   'mailing_address_str_number', 'mailing_address_str_name'],
                  inplace=True, axis=1)
```

Looking at data types, with some quick examination of non-numeric variables.


```python
training_data.dtypes
```




    ticket_id                  int64
    agency_name               object
    inspector_name            object
    violator_name             object
    ticket_issued_date        object
    hearing_date              object
    violation_code            object
    violation_description     object
    disposition               object
    fine_amount              float64
    admin_fee                float64
    state_fee                float64
    late_fee                 float64
    discount_amount          float64
    clean_up_cost            float64
    judgment_amount          float64
    compliance               float64
    lat                      float64
    lon                      float64
    violadd_equal_mailadd      uint8
    dtype: object



I see that violation description can be removed whilst date variables must be converted to a datetime format and can be merged to one variable. Also noting that there are NaN hearing dates, I have replaced these day count differences with the mean day difference of the whole dataset.


```python
training_data.drop('violation_description', inplace=True, axis=1)
training_data['ticket_issued_date'] = pd.to_datetime(training_data['ticket_issued_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')
training_data['hearing_date'] = pd.to_datetime(training_data['hearing_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')

x = (training_data['hearing_date']-training_data['ticket_issued_date']).dt.days
xmean = x[x>0].mean()
x[x.isnull()] = xmean
training_data['days_from_issue_to_hearing'] = x
training_data.drop(['hearing_date'], inplace=True, axis=1)

testing_data.drop('violation_description', inplace=True, axis=1)
testing_data['ticket_issued_date'] = pd.to_datetime(testing_data['ticket_issued_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')
testing_data['hearing_date'] = pd.to_datetime(testing_data['hearing_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')

x = (testing_data['hearing_date']-testing_data['ticket_issued_date']).dt.days
xmean = x[x>0].mean()
x[x.isnull()] = xmean
testing_data['days_from_issue_to_hearing'] = x
testing_data.drop(['hearing_date'], inplace=True, axis=1)
```

Next I look to create a new variable from violator_name which indicates whether the violator has a previous record or is a first time offender. This for-loop can take a long time to run due to the size of the datasets.


```python
training_data['prev_offence'] = np.nan
training_data = training_data.sort_values(by='ticket_issued_date')

for i in range(0, len(training_data)):
    name = training_data.iloc[i]['violator_name']
    all_names = list(training_data.iloc[:i]['violator_name'])
    if name in all_names :
        training_data.iloc[i, training_data.columns.get_loc('prev_offence')] = 1
    else :
        training_data.iloc[i, training_data.columns.get_loc('prev_offence')] = 0
        
testing_data['prev_offence'] = np.nan
testing_data = testing_data.sort_values(by='ticket_issued_date')
training_names = training_data['violator_name'].unique()

for i in range(0, len(testing_data)):
    name = testing_data.iloc[i]['violator_name']
    all_names = list(testing_data.iloc[:i]['violator_name'].unique())
    if name in all_names :
        testing_data.iloc[i, testing_data.columns.get_loc('prev_offence')] = 1
    elif name in training_names :
        testing_data.iloc[i, testing_data.columns.get_loc('prev_offence')] = 1
    else :
        testing_data.iloc[i, testing_data.columns.get_loc('prev_offence')] = 0
```

Saving the dataset for quicker loading.


```python
training_data.to_csv('training_data2.csv')
testing_data.to_csv('testing_data2.csv')
```


```python
training_data2 = pd.read_csv('training_data2.csv')
testing_data2 = pd.read_csv('testing_data2.csv')
```

Dropping unnecessary variables and noting that clean_up_cost, state_fee and admin_fee all contain the same values so can be removed. Finally, setting ticket_id as the index.


```python
training_data2.drop(['inspector_name', 'violator_name', 'ticket_issued_date'], 
                    inplace=True, axis=1)
testing_data2.drop(['inspector_name', 'violator_name', 'ticket_issued_date'], 
                    inplace=True, axis=1)
training_data2.drop(['clean_up_cost', 'admin_fee', 'state_fee', 'Unnamed: 0'], 
                    inplace=True, axis=1)
training_data2.set_index('ticket_id', inplace=True)
testing_data2.drop(['clean_up_cost', 'admin_fee', 'state_fee', 'Unnamed: 0'], 
                    inplace=True, axis=1)
testing_data2.set_index('ticket_id', inplace=True)
```

Disposition and agency_name variables are categorical so can be converted into sets of binomial variables respectively.


```python
training_data2 = pd.get_dummies(training_data2, columns = ['agency_name', 'disposition'])
testing_data2 = pd.get_dummies(testing_data2, columns = ['agency_name', 'disposition'])
training_data2['disposition_Responsible - Compl/Adj by Default'] = 0
training_data2['disposition_Responsible - Compl/Adj by Determi'] = 0
```

Violation_code is a variable of interest since the type of violation could absolutely be a strong indicator of compliance. However, with too many unique values and the potential for new violation codes to be created and appear in future testing data, the variable will have to be ignored.


```python
training_data2.drop('violation_code', inplace=True, axis=1)
testing_data2.drop('violation_code', inplace=True, axis=1)
```

Removing judgment_amount as this information is captured in the other variables concerning fees.


```python
training_data2.drop('judgment_amount', inplace=True, axis=1)
testing_data2.drop('judgment_amount', inplace=True, axis=1)
```

Looking at variable correlations to see if there are any strange relationships


```python
import seaborn as sn
import matplotlib.pyplot as plt

cols = training_data2.columns.tolist()
comp = cols[3:4]
cols.pop(3)
cols = comp+cols
df = training_data2[cols].corr()

plt.figure(figsize=(20,20))

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.savefig('corrmatimg.png')
plt.show()
```


![png](output_34_0.png)


Checking for nans one last time shows the presence of some in the lat and lon variables.


```python
training_data2.isnull().sum()/len(training_data2)
```




    fine_amount                                                   0.000000
    late_fee                                                      0.000000
    discount_amount                                               0.000000
    compliance                                                    0.000000
    lat                                                           0.000013
    lon                                                           0.000013
    violadd_equal_mailadd                                         0.000000
    days_from_issue_to_hearing                                    0.000000
    prev_offence                                                  0.000000
    agency_name_Buildings, Safety Engineering & Env Department    0.000000
    agency_name_Department of Public Works                        0.000000
    agency_name_Detroit Police Department                         0.000000
    agency_name_Health Department                                 0.000000
    agency_name_Neighborhood City Halls                           0.000000
    disposition_Responsible (Fine Waived) by Deter                0.000000
    disposition_Responsible by Admission                          0.000000
    disposition_Responsible by Default                            0.000000
    disposition_Responsible by Determination                      0.000000
    disposition_Responsible - Compl/Adj by Default                0.000000
    disposition_Responsible - Compl/Adj by Determi                0.000000
    dtype: float64



Replacing nan occurrences in lat and lon variables with the dataset mean.


```python
latmean = np.mean(training_data2['lat'])
lonmean = np.mean(training_data2['lon'])
nanlatlon = list(training_data2[training_data2['lat'].isnull()].index)
for i in nanlatlon:
    training_data2.loc[i, 'lat'] = latmean
    training_data2.loc[i, 'lon'] = lonmean

nanlatlon = list(testing_data2[testing_data2['lat'].isnull()].index)
for i in nanlatlon:
    testing_data2.loc[i, 'lat'] = latmean
    testing_data2.loc[i, 'lon'] = lonmean
```

The data has been processed to the point of including only variables that will be used in the prediction algorithms whilst having dealt with all null value occurrences.

### Prediction Models

First I will split the training data set into further training and testing sets to measure the performance of the models.


```python
from sklearn.model_selection import train_test_split

traindata = training_data2.drop('compliance', axis=1)
traincompliance = training_data2['compliance']

X_train, X_test, y_train, y_test = train_test_split(traindata, traincompliance,
                                                    random_state=0)
```

The first model I will try will be a Naive Bayes classifier as this will function as a baseline performance score that other models will be expected to beat.


```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

nbclf = GaussianNB().fit(X_train, y_train)
cross_val_score(nbclf, X_test, y_test, cv=5, scoring = 'roc_auc')
```




    array([ 0.76281538,  0.77621997,  0.77689392,  0.77946438,  0.78377937])



The next model I will try will be a Random Forest regressor (not classifier as the output has to be a probability rather than a classification).


```python
from sklearn.ensemble import RandomForestRegressor

rfclf = RandomForestRegressor(random_state=0).fit(X_train, y_train)
cross_val_score(rfclf, X_test, y_test, cv=5, scoring = 'roc_auc')
```




    array([ 0.7497346 ,  0.74796527,  0.7386544 ,  0.75410886,  0.74572621])



The performance here is surprisingly worse than that of the Naive Bayes classifier. Parameter adjustment could deal with this. However let's first try another model.

Next, I will fit a Gradient Boosted Decision Trees model.


```python
from sklearn.ensemble import GradientBoostingRegressor

gbclf = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
cross_val_score(gbclf, X_test, y_test, cv=5, scoring = 'roc_auc')
```




    array([ 0.80726241,  0.80475678,  0.81073189,  0.81416076,  0.82483667])



With this good performance I will proceed further with this model and see if the parameters can be adjusted to increase accuracy and/or reduce overfitting.

In order to do so I will use the GridSearch function noting that whilst ideally wider ranges for the parameters would be tested and that n_estimators and learning_rate would be tested with max_depth in tandem, computer processing speed and time constraints require the process to be limited and divided.


```python
from sklearn.model_selection import GridSearchCV
grid_values = {'n_estimators': [50,75,100], 
               'learning_rate': [0.01, 0.1, 0.3, 0.4]}
grid_gbclf = GridSearchCV(gbclf, param_grid = grid_values, scoring='roc_auc')
grid_gbclf.fit(X_train, y_train)
print(grid_gbclf.best_params_)
print(grid_gbclf.best_score_)
```

    {'learning_rate': 0.3, 'n_estimators': 50}
    0.820923386309


Now testing with these parameters.


```python
gbclf = GradientBoostingRegressor(random_state=0,
                                  n_estimators=50,
                                  learning_rate=0.3).fit(X_train, y_train)
cross_val_score(gbclf, X_test, y_test, cv=5, scoring = 'roc_auc')
```




    array([ 0.8005826 ,  0.80654677,  0.81217829,  0.81221508,  0.81939989])



Performance doesn't seem to have changed too much. Now, finding the best max_depth parameter.


```python
grid_values = {'max_depth': [3,4,5]}
grid_gbclf = GridSearchCV(gbclf, param_grid = grid_values, scoring='roc_auc')
grid_gbclf.fit(X_train, y_train)
print(grid_gbclf.best_params_)
print(grid_gbclf.best_score_)
```

    {'max_depth': 5}
    0.82173473978


Now testing with this parameter too.


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
gbclf = GradientBoostingRegressor(random_state=0,
                                  n_estimators=50,
                                  learning_rate=0.3,
                                 max_depth=5).fit(X_train, y_train)
cross_val_score(gbclf, X_test, y_test, cv=5, scoring = 'roc_auc')
```




    array([ 0.80116443,  0.80810559,  0.80593239,  0.80971667,  0.80993976])



Again, the performance doesn't seem to have improved as such, but the chance of reduced overfitting is improved and thus I will proceed with this model as my final choice. Below I print it's average performance.


```python
np.mean([0.80116443,  0.80810559,  0.80593239,  0.80971667,  0.80993976])
```




    0.80697176800000003



Now, putting the final predictions into the required output format.


```python
y_preds = gbclf.predict(testing_data2)
y_preds = pd.Series(y_preds, index=list(testing_data2.index))
y_preds = y_preds.rename_axis('ticket_id')
y_preds.head()
```




    ticket_id
    284932    0.168202
    285343    0.144434
    285344    0.365637
    285362    0.120585
    285342    0.880129
    dtype: float64



### Submission

Due to the restraints of the auto-grader system, the function I submitted was rejected. The original function I submitted included all of the code as written above but, with a total runtime of about 25 minutes, it surpasses the time limit of the auto-grader system.

As such, I had to remove the prev_offence variable that I had created as that is what eats up most of the time when the whole code is run. Since the dataset changes with this removal, with more time, I could explore further parameter adjustment in the Gradient Boosted model and certainly see if I can get more performance out of the Random Trees regressor. However, since the fact that prior to this removal the model performs at a much higher level than the required 0.75 AUC score, I will just proceed with the same model as previously written.


```python
def blight_model():
    
    training_data = pd.read_csv('readonly/train.csv', encoding='latin1')
    testing_data = pd.read_csv('readonly/test.csv')
    address_data = pd.read_csv('readonly/addresses.csv')
    latlon_data = pd.read_csv('readonly/latlons.csv')
    
    unavailable_vars = ['payment_amount', 'payment_date', 'payment_status', 
                   'balance_due', 'collection_status', 'compliance_detail']
    training_data.drop(unavailable_vars, inplace=True, axis=1)
    
    training_data.drop(['violation_zip_code', 'grafitti_status', 'non_us_str_code',
                       'city', 'state', 'zip_code', 'country'], 
                   inplace=True, axis=1)
    testing_data.drop(['violation_zip_code', 'grafitti_status', 'non_us_str_code',
                      'city', 'state', 'zip_code', 'country'], 
                   inplace=True, axis=1)
    
    training_data = training_data.loc[~training_data['compliance'].isnull()]
    
    addlanlon_data = address_data.merge(latlon_data, how='inner', on='address')
    addlanlon_data.drop('address', inplace=True, axis=1)
    training_data = training_data.merge(addlanlon_data, how='inner', on='ticket_id')
    testing_data = testing_data.merge(addlanlon_data, how='inner', on='ticket_id')
    
    training_data['violadd_equal_mailadd'] = training_data['violation_street_name']==training_data['mailing_address_str_name']
    training_data['violadd_equal_mailadd'] = training_data['violadd_equal_mailadd'].astype('uint8')
    training_data.drop(['violation_street_number', 'violation_street_name',
                   'mailing_address_str_number', 'mailing_address_str_name'],
                  inplace=True, axis=1)

    testing_data['violadd_equal_mailadd'] = testing_data['violation_street_name']==testing_data['mailing_address_str_name']
    testing_data['violadd_equal_mailadd'] = testing_data['violadd_equal_mailadd'].astype('uint8')
    testing_data.drop(['violation_street_number', 'violation_street_name',
                   'mailing_address_str_number', 'mailing_address_str_name'],
                  inplace=True, axis=1)
    
    training_data.drop('violation_description', inplace=True, axis=1)
    training_data['ticket_issued_date'] = pd.to_datetime(training_data['ticket_issued_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')
    training_data['hearing_date'] = pd.to_datetime(training_data['hearing_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')

    x = (training_data['hearing_date']-training_data['ticket_issued_date']).dt.days
    xmean = x[x>0].mean()
    x[x.isnull()] = xmean
    training_data['days_from_issue_to_hearing'] = x
    training_data.drop(['hearing_date'], inplace=True, axis=1)

    testing_data.drop('violation_description', inplace=True, axis=1)
    testing_data['ticket_issued_date'] = pd.to_datetime(testing_data['ticket_issued_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')
    testing_data['hearing_date'] = pd.to_datetime(testing_data['hearing_date'] , 
                                                     format='%Y/%m/%d %H:%M:%S')

    x = (testing_data['hearing_date']-testing_data['ticket_issued_date']).dt.days
    xmean = x[x>0].mean()
    x[x.isnull()] = xmean
    testing_data['days_from_issue_to_hearing'] = x
    testing_data.drop(['hearing_date'], inplace=True, axis=1)
    
    training_data.drop(['inspector_name', 'violator_name', 'ticket_issued_date'], 
                    inplace=True, axis=1)
    testing_data.drop(['inspector_name', 'violator_name', 'ticket_issued_date'], 
                    inplace=True, axis=1)

    training_data.drop(['clean_up_cost', 'admin_fee', 'state_fee'], 
                    inplace=True, axis=1)
    training_data.set_index('ticket_id', inplace=True)
    testing_data.drop(['clean_up_cost', 'admin_fee', 'state_fee'], 
                    inplace=True, axis=1)
    testing_data.set_index('ticket_id', inplace=True)
    
    training_data = pd.get_dummies(training_data, columns = ['agency_name', 'disposition'])
    testing_data = pd.get_dummies(testing_data, columns = ['agency_name', 'disposition'])
    training_data['disposition_Responsible - Compl/Adj by Default'] = 0
    training_data['disposition_Responsible - Compl/Adj by Determi'] = 0
    
    training_data.drop(['violation_code', 'judgment_amount'], inplace=True, axis=1)
    testing_data.drop(['violation_code', 'judgment_amount'], inplace=True, axis=1)
    
    latmean = np.mean(training_data['lat'])
    lonmean = np.mean(training_data['lon'])
    nanlatlon = list(training_data[training_data['lat'].isnull()].index)
    for i in nanlatlon:
        training_data.loc[i, 'lat'] = latmean
        training_data.loc[i, 'lon'] = lonmean

    nanlatlon = list(testing_data[testing_data['lat'].isnull()].index)
    for i in nanlatlon:
        testing_data.loc[i, 'lat'] = latmean
        testing_data.loc[i, 'lon'] = lonmean
        
    from sklearn.model_selection import train_test_split

    traindata = training_data.drop('compliance', axis=1)
    traincompliance = training_data['compliance']
    
    from sklearn.ensemble import GradientBoostingRegressor

    X_train, X_test, y_train, y_test = train_test_split(traindata, traincompliance,
                                                    random_state=0)
    gbclf = GradientBoostingRegressor(random_state=0,
                                  n_estimators=50,
                                  learning_rate=0.3,
                                  max_depth=5).fit(X_train, y_train)
    
    y_preds = gbclf.predict(testing_data)
    y_preds = pd.Series(y_preds, index=list(testing_data.index))
    y_preds = y_preds.rename_axis('ticket_id')
    
    return y_preds
```


```python
blight_model()
```




    ticket_id
    284932    0.064211
    285362    0.155713
    285361    0.237549
    285338    0.356996
    285346    0.287712
    285345    0.281238
    285347    0.293473
    285342    0.906370
    285530    0.098729
    284989    0.200572
    285344    0.330670
    285343    0.134529
    285340    0.085863
    285341    0.293473
    285349    0.287712
    285348    0.281238
    284991    0.200572
    285532    0.090584
    285406    0.192021
    285001    0.314366
    285006    0.237704
    285405    0.155713
    285337    0.182413
    285496    0.290411
    285497    0.303337
    285378    0.159575
    285589    0.186106
    285585    0.235476
    285501    0.198595
    285581    0.156410
                ...   
    376367    0.162088
    376366    0.233052
    376362    0.223281
    376363    0.223281
    376365    0.162088
    376364    0.233052
    376228    0.307893
    376265    0.227728
    376286    0.593699
    376320    0.236003
    376314    0.239244
    376327    0.775396
    376385    0.876933
    376435    0.928445
    376370    0.884618
    376434    0.249711
    376459    0.287269
    376478    0.108850
    376473    0.240179
    376484    0.200913
    376482    0.192501
    376480    0.192501
    376479    0.192501
    376481    0.192501
    376483    0.233817
    376496    0.114372
    376497    0.114372
    376499    0.221896
    376500    0.227796
    369851    0.900557
    dtype: float64



According to the auto-grader system:

***Your AUC of 0.782910006428 was awarded a value of 1.0 out of 1.0 total grades.***


### End

Thank you to the professors at University of Michigan for the interesting and insightful educational experience.

[https://github.com/jamesk185](https://github.com/jamesk185)


```python

```
