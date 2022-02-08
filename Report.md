
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
cols = training_data2.columns.tolist()
comp = cols[3:4]
cols.pop(3)
cols = comp+cols
corr = training_data2[cols].corr()
corr.style.background_gradient(cmap='coolwarm')
```





        <style  type="text/css" >
        
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col0 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col1 {
            
                background-color:  #5b7ae5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col2 {
            
                background-color:  #516ddb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col3 {
            
                background-color:  #86a9fc;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col4 {
            
                background-color:  #4257c9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col5 {
            
                background-color:  #4c66d6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col6 {
            
                background-color:  #5e7de7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col7 {
            
                background-color:  #4c66d6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col8 {
            
                background-color:  #4961d2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col9 {
            
                background-color:  #cad8ef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col10 {
            
                background-color:  #d9dce1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col11 {
            
                background-color:  #7da0f9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col12 {
            
                background-color:  #7da0f9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col14 {
            
                background-color:  #7a9df8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col15 {
            
                background-color:  #edd1c2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col16 {
            
                background-color:  #8caffe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col17 {
            
                background-color:  #dadce0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col0 {
            
                background-color:  #80a3fa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col1 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col2 {
            
                background-color:  #b70d28;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col3 {
            
                background-color:  #6180e9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col4 {
            
                background-color:  #485fd1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col5 {
            
                background-color:  #465ecf;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col6 {
            
                background-color:  #536edd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col7 {
            
                background-color:  #465ecf;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col8 {
            
                background-color:  #6485ec;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col9 {
            
                background-color:  #b5cdfa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col10 {
            
                background-color:  #edd1c2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col11 {
            
                background-color:  #6485ec;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col12 {
            
                background-color:  #779af7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col13 {
            
                background-color:  #3d50c3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col14 {
            
                background-color:  #4e68d8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col15 {
            
                background-color:  #c0d4f5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col16 {
            
                background-color:  #d2dbe8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col17 {
            
                background-color:  #b9d0f9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col0 {
            
                background-color:  #779af7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col1 {
            
                background-color:  #b70d28;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col2 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col3 {
            
                background-color:  #4f69d9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col4 {
            
                background-color:  #4961d2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col5 {
            
                background-color:  #465ecf;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col6 {
            
                background-color:  #5470de;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col7 {
            
                background-color:  #4961d2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col8 {
            
                background-color:  #6788ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col9 {
            
                background-color:  #b6cefa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col10 {
            
                background-color:  #edd1c2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col11 {
            
                background-color:  #6687ed;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col12 {
            
                background-color:  #779af7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col13 {
            
                background-color:  #3d50c3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col14 {
            
                background-color:  #4e68d8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col15 {
            
                background-color:  #b1cbfc;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col16 {
            
                background-color:  #dcdddd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col17 {
            
                background-color:  #b7cff9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col0 {
            
                background-color:  #b6cefa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col1 {
            
                background-color:  #7597f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col2 {
            
                background-color:  #6282ea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col3 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col4 {
            
                background-color:  #4961d2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col5 {
            
                background-color:  #4b64d5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col6 {
            
                background-color:  #5b7ae5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col7 {
            
                background-color:  #5e7de7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col8 {
            
                background-color:  #6180e9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col9 {
            
                background-color:  #d1dae9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col10 {
            
                background-color:  #d4dbe6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col11 {
            
                background-color:  #7093f3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col12 {
            
                background-color:  #7ea1fa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col14 {
            
                background-color:  #536edd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col15 {
            
                background-color:  #dddcdc;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col16 {
            
                background-color:  #bad0f8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col17 {
            
                background-color:  #b3cdfb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col0 {
            
                background-color:  #88abfd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col1 {
            
                background-color:  #688aef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col2 {
            
                background-color:  #688aef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col3 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col4 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col5 {
            
                background-color:  #85a8fc;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col6 {
            
                background-color:  #5977e3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col7 {
            
                background-color:  #5673e0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col8 {
            
                background-color:  #688aef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col9 {
            
                background-color:  #cbd8ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col10 {
            
                background-color:  #dbdcde;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col11 {
            
                background-color:  #6788ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col12 {
            
                background-color:  #80a3fa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col13 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col14 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col15 {
            
                background-color:  #c6d6f1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col16 {
            
                background-color:  #cfdaea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col17 {
            
                background-color:  #b3cdfb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col0 {
            
                background-color:  #8db0fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col1 {
            
                background-color:  #6282ea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col2 {
            
                background-color:  #6282ea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col3 {
            
                background-color:  #5470de;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col4 {
            
                background-color:  #80a3fa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col5 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col6 {
            
                background-color:  #5b7ae5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col7 {
            
                background-color:  #5875e1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col8 {
            
                background-color:  #6687ed;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col9 {
            
                background-color:  #dcdddd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col10 {
            
                background-color:  #ccd9ed;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col11 {
            
                background-color:  #5e7de7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col12 {
            
                background-color:  #7699f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col13 {
            
                background-color:  #3d50c3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col14 {
            
                background-color:  #516ddb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col15 {
            
                background-color:  #c9d7f0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col16 {
            
                background-color:  #cedaeb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col17 {
            
                background-color:  #b3cdfb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col0 {
            
                background-color:  #90b2fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col1 {
            
                background-color:  #6180e9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col2 {
            
                background-color:  #6180e9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col3 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col4 {
            
                background-color:  #455cce;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col5 {
            
                background-color:  #4c66d6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col6 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col7 {
            
                background-color:  #4b64d5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col8 {
            
                background-color:  #485fd1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col9 {
            
                background-color:  #bfd3f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col10 {
            
                background-color:  #e1dad6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col11 {
            
                background-color:  #6e90f2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col12 {
            
                background-color:  #92b4fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col14 {
            
                background-color:  #5977e3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col15 {
            
                background-color:  #cbd8ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col16 {
            
                background-color:  #c7d7f0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col17 {
            
                background-color:  #bcd2f7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col0 {
            
                background-color:  #8caffe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col1 {
            
                background-color:  #6384eb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col2 {
            
                background-color:  #6485ec;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col3 {
            
                background-color:  #6788ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col4 {
            
                background-color:  #516ddb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col5 {
            
                background-color:  #5875e1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col6 {
            
                background-color:  #5977e3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col7 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col8 {
            
                background-color:  #6384eb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col9 {
            
                background-color:  #dbdcde;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col10 {
            
                background-color:  #cbd8ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col11 {
            
                background-color:  #6384eb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col12 {
            
                background-color:  #80a3fa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col13 {
            
                background-color:  #3e51c5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col14 {
            
                background-color:  #536edd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col15 {
            
                background-color:  #c0d4f5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col16 {
            
                background-color:  #d4dbe6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col17 {
            
                background-color:  #b6cefa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col0 {
            
                background-color:  #7597f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col1 {
            
                background-color:  #6a8bef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col2 {
            
                background-color:  #6c8ff1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col3 {
            
                background-color:  #536edd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col4 {
            
                background-color:  #4c66d6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col5 {
            
                background-color:  #506bda;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col6 {
            
                background-color:  #3f53c6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col7 {
            
                background-color:  #4c66d6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col8 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col9 {
            
                background-color:  #eed0c0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col10 {
            
                background-color:  #bad0f8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col11 {
            
                background-color:  #5875e1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col12 {
            
                background-color:  #6384eb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col14 {
            
                background-color:  #4f69d9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col15 {
            
                background-color:  #b9d0f9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col16 {
            
                background-color:  #e0dbd8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col17 {
            
                background-color:  #a2c1ff;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col0 {
            
                background-color:  #7ea1fa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col1 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col2 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col3 {
            
                background-color:  #536edd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col4 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col5 {
            
                background-color:  #6282ea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col6 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col7 {
            
                background-color:  #5f7fe8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col8 {
            
                background-color:  #a6c4fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col9 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col10 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col11 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col12 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col13 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col14 {
            
                background-color:  #485fd1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col15 {
            
                background-color:  #c3d5f4;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col16 {
            
                background-color:  #d9dce1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col17 {
            
                background-color:  #a5c3fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col0 {
            
                background-color:  #9abbff;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col1 {
            
                background-color:  #a7c5fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col2 {
            
                background-color:  #a7c5fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col3 {
            
                background-color:  #5875e1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col4 {
            
                background-color:  #5b7ae5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col5 {
            
                background-color:  #4257c9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col6 {
            
                background-color:  #7b9ff9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col7 {
            
                background-color:  #3f53c6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col8 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col9 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col10 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col11 {
            
                background-color:  #506bda;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col12 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col14 {
            
                background-color:  #5f7fe8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col15 {
            
                background-color:  #cfdaea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col16 {
            
                background-color:  #c0d4f5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col17 {
            
                background-color:  #c0d4f5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col0 {
            
                background-color:  #97b8ff;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col1 {
            
                background-color:  #5d7ce6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col2 {
            
                background-color:  #5d7ce6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col3 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col4 {
            
                background-color:  #3e51c5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col5 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col6 {
            
                background-color:  #5875e1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col7 {
            
                background-color:  #3f53c6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col8 {
            
                background-color:  #4a63d3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col9 {
            
                background-color:  #afcafc;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col10 {
            
                background-color:  #bfd3f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col11 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col12 {
            
                background-color:  #7597f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col14 {
            
                background-color:  #5875e1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col15 {
            
                background-color:  #d2dbe8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col16 {
            
                background-color:  #c4d5f3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col17 {
            
                background-color:  #b7cff9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col0 {
            
                background-color:  #8caffe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col1 {
            
                background-color:  #6282ea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col2 {
            
                background-color:  #6282ea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col3 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col4 {
            
                background-color:  #4961d2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col5 {
            
                background-color:  #455cce;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col6 {
            
                background-color:  #7093f3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col7 {
            
                background-color:  #4f69d9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col8 {
            
                background-color:  #485fd1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col9 {
            
                background-color:  #a5c3fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col10 {
            
                background-color:  #bad0f8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col11 {
            
                background-color:  #6788ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col12 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col14 {
            
                background-color:  #516ddb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col15 {
            
                background-color:  #d1dae9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col16 {
            
                background-color:  #c5d6f2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col17 {
            
                background-color:  #b7cff9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col0 {
            
                background-color:  #8db0fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col1 {
            
                background-color:  #6a8bef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col2 {
            
                background-color:  #6a8bef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col3 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col4 {
            
                background-color:  #485fd1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col5 {
            
                background-color:  #4e68d8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col6 {
            
                background-color:  #5b7ae5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col7 {
            
                background-color:  #506bda;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col8 {
            
                background-color:  #6384eb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col9 {
            
                background-color:  #d2dbe8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col10 {
            
                background-color:  #d2dbe8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col11 {
            
                background-color:  #7093f3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col12 {
            
                background-color:  #7ea1fa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col13 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col14 {
            
                background-color:  #536edd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col15 {
            
                background-color:  #cbd8ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col16 {
            
                background-color:  #cad8ef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col17 {
            
                background-color:  #b6cefa;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col0 {
            
                background-color:  #aec9fc;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col1 {
            
                background-color:  #6485ec;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col2 {
            
                background-color:  #6485ec;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col3 {
            
                background-color:  #5572df;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col4 {
            
                background-color:  #4a63d3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col5 {
            
                background-color:  #4b64d5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col6 {
            
                background-color:  #6282ea;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col7 {
            
                background-color:  #4c66d6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col8 {
            
                background-color:  #5f7fe8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col9 {
            
                background-color:  #ccd9ed;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col10 {
            
                background-color:  #d8dce2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col11 {
            
                background-color:  #7699f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col12 {
            
                background-color:  #7b9ff9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col14 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col15 {
            
                background-color:  #cad8ef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col16 {
            
                background-color:  #bcd2f7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col17 {
            
                background-color:  #b2ccfb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col0 {
            
                background-color:  #cad8ef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col1 {
            
                background-color:  #5875e1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col2 {
            
                background-color:  #4257c9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col3 {
            
                background-color:  #7a9df8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col4 {
            
                background-color:  #4055c8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col5 {
            
                background-color:  #485fd1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col6 {
            
                background-color:  #5b7ae5;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col7 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col8 {
            
                background-color:  #465ecf;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col9 {
            
                background-color:  #cad8ef;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col10 {
            
                background-color:  #d6dce4;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col11 {
            
                background-color:  #7da0f9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col12 {
            
                background-color:  #86a9fc;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col13 {
            
                background-color:  #3c4ec2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col14 {
            
                background-color:  #506bda;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col15 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col16 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col17 {
            
                background-color:  #a5c3fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col0 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col1 {
            
                background-color:  #7597f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col2 {
            
                background-color:  #8badfd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col3 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col4 {
            
                background-color:  #4f69d9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col5 {
            
                background-color:  #536edd;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col6 {
            
                background-color:  #5470de;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col7 {
            
                background-color:  #5d7ce6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col8 {
            
                background-color:  #8db0fe;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col9 {
            
                background-color:  #dfdbd9;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col10 {
            
                background-color:  #c9d7f0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col11 {
            
                background-color:  #6485ec;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col12 {
            
                background-color:  #7597f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col13 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col14 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col15 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col16 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col17 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col0 {
            
                background-color:  #c1d4f4;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col1 {
            
                background-color:  #7093f3;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col2 {
            
                background-color:  #6e90f2;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col3 {
            
                background-color:  #5673e0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col4 {
            
                background-color:  #485fd1;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col5 {
            
                background-color:  #4c66d6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col6 {
            
                background-color:  #6788ee;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col7 {
            
                background-color:  #506bda;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col8 {
            
                background-color:  #4e68d8;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col9 {
            
                background-color:  #c7d7f0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col10 {
            
                background-color:  #dbdcde;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col11 {
            
                background-color:  #7699f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col12 {
            
                background-color:  #82a6fb;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col13 {
            
                background-color:  #3f53c6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col14 {
            
                background-color:  #506bda;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col15 {
            
                background-color:  #bfd3f6;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col16 {
            
                background-color:  #5e7de7;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col17 {
            
                background-color:  #b40426;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col0 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col1 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col2 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col3 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col4 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col5 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col6 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col7 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col8 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col9 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col10 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col11 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col12 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col13 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col14 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col15 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col16 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col17 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col0 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col1 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col2 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col3 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col4 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col5 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col6 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col7 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col8 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col9 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col10 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col11 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col12 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col13 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col14 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col15 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col16 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col17 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col18 {
            
                background-color:  #3b4cc0;
            
            }
        
            #T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col19 {
            
                background-color:  #3b4cc0;
            
            }
        
        </style>

        <table id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085" None>
        

        <thead>
            
            <tr>
                
                
                <th class="blank level0" >
                  
                
                
                
                <th class="col_heading level0 col0" colspan=1>
                  compliance
                
                
                
                <th class="col_heading level0 col1" colspan=1>
                  fine_amount
                
                
                
                <th class="col_heading level0 col2" colspan=1>
                  late_fee
                
                
                
                <th class="col_heading level0 col3" colspan=1>
                  discount_amount
                
                
                
                <th class="col_heading level0 col4" colspan=1>
                  lat
                
                
                
                <th class="col_heading level0 col5" colspan=1>
                  lon
                
                
                
                <th class="col_heading level0 col6" colspan=1>
                  violadd_equal_mailadd
                
                
                
                <th class="col_heading level0 col7" colspan=1>
                  days_from_issue_to_hearing
                
                
                
                <th class="col_heading level0 col8" colspan=1>
                  prev_offence
                
                
                
                <th class="col_heading level0 col9" colspan=1>
                  agency_name_Buildings, Safety Engineering & Env Department
                
                
                
                <th class="col_heading level0 col10" colspan=1>
                  agency_name_Department of Public Works
                
                
                
                <th class="col_heading level0 col11" colspan=1>
                  agency_name_Detroit Police Department
                
                
                
                <th class="col_heading level0 col12" colspan=1>
                  agency_name_Health Department
                
                
                
                <th class="col_heading level0 col13" colspan=1>
                  agency_name_Neighborhood City Halls
                
                
                
                <th class="col_heading level0 col14" colspan=1>
                  disposition_Responsible (Fine Waived) by Deter
                
                
                
                <th class="col_heading level0 col15" colspan=1>
                  disposition_Responsible by Admission
                
                
                
                <th class="col_heading level0 col16" colspan=1>
                  disposition_Responsible by Default
                
                
                
                <th class="col_heading level0 col17" colspan=1>
                  disposition_Responsible by Determination
                
                
                
                <th class="col_heading level0 col18" colspan=1>
                  disposition_Responsible - Compl/Adj by Default
                
                
                
                <th class="col_heading level0 col19" colspan=1>
                  disposition_Responsible - Compl/Adj by Determi
                
                
            </tr>
            
        </thead>
        <tbody>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row0" rowspan=1>
                    compliance
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col0"
                 class="data row0 col0" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col1"
                 class="data row0 col1" >
                    -0.0491341
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col2"
                 class="data row0 col2" >
                    -0.0850553
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col3"
                 class="data row0 col3" >
                    0.156073
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col4"
                 class="data row0 col4" >
                    -0.0215693
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col5"
                 class="data row0 col5" >
                    -0.000430988
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col6"
                 class="data row0 col6" >
                    0.00893281
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col7"
                 class="data row0 col7" >
                    -0.00466586
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col8"
                 class="data row0 col8" >
                    -0.0933103
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col9"
                 class="data row0 col9" >
                    -0.0556375
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col10"
                 class="data row0 col10" >
                    0.046939
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col11"
                 class="data row0 col11" >
                    0.0386723
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col12"
                 class="data row0 col12" >
                    -0.00555907
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col13"
                 class="data row0 col13" >
                    -0.000699409
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col14"
                 class="data row0 col14" >
                    0.124956
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col15"
                 class="data row0 col15" >
                    0.238997
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col16"
                 class="data row0 col16" >
                    -0.335455
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col17"
                 class="data row0 col17" >
                    0.202819
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col18"
                 class="data row0 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row0_col19"
                 class="data row0 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row1" rowspan=1>
                    fine_amount
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col0"
                 class="data row1 col0" >
                    -0.0491341
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col1"
                 class="data row1 col1" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col2"
                 class="data row1 col2" >
                    0.986787
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col3"
                 class="data row1 col3" >
                    0.0394948
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col4"
                 class="data row1 col4" >
                    -0.00289098
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col5"
                 class="data row1 col5" >
                    -0.0240704
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col6"
                 class="data row1 col6" >
                    -0.0315075
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col7"
                 class="data row1 col7" >
                    -0.0232237
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col8"
                 class="data row1 col8" >
                    0.0024617
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col9"
                 class="data row1 col9" >
                    -0.175369
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col10"
                 class="data row1 col10" >
                    0.209533
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col11"
                 class="data row1 col11" >
                    -0.0424905
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col12"
                 class="data row1 col12" >
                    -0.0264583
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col13"
                 class="data row1 col13" >
                    0.00237995
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col14"
                 class="data row1 col14" >
                    -0.018466
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col15"
                 class="data row1 col15" >
                    -0.0640385
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col16"
                 class="data row1 col16" >
                    0.0380344
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col17"
                 class="data row1 col17" >
                    0.0261695
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col18"
                 class="data row1 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row1_col19"
                 class="data row1 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row2" rowspan=1>
                    late_fee
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col0"
                 class="data row2 col0" >
                    -0.0850553
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col1"
                 class="data row2 col1" >
                    0.986787
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col2"
                 class="data row2 col2" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col3"
                 class="data row2 col3" >
                    -0.0227059
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col4"
                 class="data row2 col4" >
                    0.000513961
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col5"
                 class="data row2 col5" >
                    -0.0227273
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col6"
                 class="data row2 col6" >
                    -0.0292747
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col7"
                 class="data row2 col7" >
                    -0.0172742
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col8"
                 class="data row2 col8" >
                    0.0107697
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col9"
                 class="data row2 col9" >
                    -0.173658
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col10"
                 class="data row2 col10" >
                    0.206868
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col11"
                 class="data row2 col11" >
                    -0.0418958
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col12"
                 class="data row2 col12" >
                    -0.024933
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col13"
                 class="data row2 col13" >
                    0.00245128
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col14"
                 class="data row2 col14" >
                    -0.0173719
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col15"
                 class="data row2 col15" >
                    -0.145608
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col16"
                 class="data row2 col16" >
                    0.111337
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col17"
                 class="data row2 col17" >
                    0.0157017
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col18"
                 class="data row2 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row2_col19"
                 class="data row2 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row3" rowspan=1>
                    discount_amount
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col0"
                 class="data row3 col0" >
                    0.156073
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col1"
                 class="data row3 col1" >
                    0.0394948
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col2"
                 class="data row3 col2" >
                    -0.0227059
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col3"
                 class="data row3 col3" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col4"
                 class="data row3 col4" >
                    0.000261185
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col5"
                 class="data row3 col5" >
                    -0.00643173
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col6"
                 class="data row3 col6" >
                    -0.000768894
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col7"
                 class="data row3 col7" >
                    0.0573549
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col8"
                 class="data row3 col8" >
                    -0.00776834
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col9"
                 class="data row3 col9" >
                    -0.0085519
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col10"
                 class="data row3 col10" >
                    0.00880595
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col11"
                 class="data row3 col11" >
                    -0.000785474
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col12"
                 class="data row3 col12" >
                    0.000900588
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col13"
                 class="data row3 col13" >
                    -0.00011423
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col14"
                 class="data row3 col14" >
                    -0.0015961
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col15"
                 class="data row3 col15" >
                    0.11791
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col16"
                 class="data row3 col16" >
                    -0.0973869
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col17"
                 class="data row3 col17" >
                    0.00141495
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col18"
                 class="data row3 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row3_col19"
                 class="data row3 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row4" rowspan=1>
                    lat
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col0"
                 class="data row4 col0" >
                    -0.0215693
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col1"
                 class="data row4 col1" >
                    -0.00289098
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col2"
                 class="data row4 col2" >
                    0.000513961
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col3"
                 class="data row4 col3" >
                    0.000261185
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col4"
                 class="data row4 col4" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col5"
                 class="data row4 col5" >
                    0.175336
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col6"
                 class="data row4 col6" >
                    -0.00918752
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col7"
                 class="data row4 col7" >
                    0.0290633
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col8"
                 class="data row4 col8" >
                    0.0159037
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col9"
                 class="data row4 col9" >
                    -0.0495605
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col10"
                 class="data row4 col10" >
                    0.0629748
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col11"
                 class="data row4 col11" >
                    -0.035747
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col12"
                 class="data row4 col12" >
                    0.00299099
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col13"
                 class="data row4 col13" >
                    -0.00393082
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col14"
                 class="data row4 col14" >
                    0.00626365
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col15"
                 class="data row4 col15" >
                    -0.0273877
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col16"
                 class="data row4 col16" >
                    0.0242223
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col17"
                 class="data row4 col17" >
                    -0.00385557
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col18"
                 class="data row4 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row4_col19"
                 class="data row4 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row5" rowspan=1>
                    lon
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col0"
                 class="data row5 col0" >
                    -0.000430988
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col1"
                 class="data row5 col1" >
                    -0.0240704
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col2"
                 class="data row5 col2" >
                    -0.0227273
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col3"
                 class="data row5 col3" >
                    -0.00643173
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col4"
                 class="data row5 col4" >
                    0.175336
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col5"
                 class="data row5 col5" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col6"
                 class="data row5 col6" >
                    -0.00194438
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col7"
                 class="data row5 col7" >
                    0.0356799
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col8"
                 class="data row5 col8" >
                    0.00889877
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col9"
                 class="data row5 col9" >
                    0.0704376
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col10"
                 class="data row5 col10" >
                    -0.0379802
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col11"
                 class="data row5 col11" >
                    -0.0665971
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col12"
                 class="data row5 col12" >
                    -0.0277263
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col13"
                 class="data row5 col13" >
                    0.0017902
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col14"
                 class="data row5 col14" >
                    -0.00736269
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col15"
                 class="data row5 col15" >
                    -0.0193673
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col16"
                 class="data row5 col16" >
                    0.0175426
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col17"
                 class="data row5 col17" >
                    -0.00145961
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col18"
                 class="data row5 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row5_col19"
                 class="data row5 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row6" rowspan=1>
                    violadd_equal_mailadd
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col0"
                 class="data row6 col0" >
                    0.00893281
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col1"
                 class="data row6 col1" >
                    -0.0315075
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col2"
                 class="data row6 col2" >
                    -0.0292747
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col3"
                 class="data row6 col3" >
                    -0.000768894
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col4"
                 class="data row6 col4" >
                    -0.00918752
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col5"
                 class="data row6 col5" >
                    -0.00194438
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col6"
                 class="data row6 col6" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col7"
                 class="data row6 col7" >
                    -0.00797271
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col8"
                 class="data row6 col8" >
                    -0.0997978
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col9"
                 class="data row6 col9" >
                    -0.121658
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col10"
                 class="data row6 col10" >
                    0.102029
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col11"
                 class="data row6 col11" >
                    -0.0140086
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col12"
                 class="data row6 col12" >
                    0.0680206
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col13"
                 class="data row6 col13" >
                    -0.00128837
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col14"
                 class="data row6 col14" >
                    0.0202941
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col15"
                 class="data row6 col15" >
                    -0.00194609
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col16"
                 class="data row6 col16" >
                    -0.0256261
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col17"
                 class="data row6 col17" >
                    0.0402406
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col18"
                 class="data row6 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row6_col19"
                 class="data row6 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row7" rowspan=1>
                    days_from_issue_to_hearing
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col0"
                 class="data row7 col0" >
                    -0.00466586
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col1"
                 class="data row7 col1" >
                    -0.0232237
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col2"
                 class="data row7 col2" >
                    -0.0172742
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col3"
                 class="data row7 col3" >
                    0.0573549
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col4"
                 class="data row7 col4" >
                    0.0290633
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col5"
                 class="data row7 col5" >
                    0.0356799
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col6"
                 class="data row7 col6" >
                    -0.00797271
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col7"
                 class="data row7 col7" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col8"
                 class="data row7 col8" >
                    -0.00102249
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col9"
                 class="data row7 col9" >
                    0.0603615
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col10"
                 class="data row7 col10" >
                    -0.0488367
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col11"
                 class="data row7 col11" >
                    -0.0484144
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col12"
                 class="data row7 col12" >
                    0.00634507
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col13"
                 class="data row7 col13" >
                    0.00840098
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col14"
                 class="data row7 col14" >
                    -0.000776949
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col15"
                 class="data row7 col15" >
                    -0.0673838
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col16"
                 class="data row7 col16" >
                    0.0497373
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col17"
                 class="data row7 col17" >
                    0.0089375
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col18"
                 class="data row7 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row7_col19"
                 class="data row7 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row8" rowspan=1>
                    prev_offence
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col0"
                 class="data row8 col0" >
                    -0.0933103
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col1"
                 class="data row8 col1" >
                    0.0024617
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col2"
                 class="data row8 col2" >
                    0.0107697
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col3"
                 class="data row8 col3" >
                    -0.00776834
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col4"
                 class="data row8 col4" >
                    0.0159037
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col5"
                 class="data row8 col5" >
                    0.00889877
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col6"
                 class="data row8 col6" >
                    -0.0997978
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col7"
                 class="data row8 col7" >
                    -0.00102249
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col8"
                 class="data row8 col8" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col9"
                 class="data row8 col9" >
                    0.217381
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col10"
                 class="data row8 col10" >
                    -0.151716
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col11"
                 class="data row8 col11" >
                    -0.0904623
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col12"
                 class="data row8 col12" >
                    -0.0988538
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col13"
                 class="data row8 col13" >
                    -0.00235748
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col14"
                 class="data row8 col14" >
                    -0.0153477
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col15"
                 class="data row8 col15" >
                    -0.105844
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col16"
                 class="data row8 col16" >
                    0.138091
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col17"
                 class="data row8 col17" >
                    -0.0796086
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col18"
                 class="data row8 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row8_col19"
                 class="data row8 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row9" rowspan=1>
                    agency_name_Buildings, Safety Engineering & Env Department
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col0"
                 class="data row9 col0" >
                    -0.0556375
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col1"
                 class="data row9 col1" >
                    -0.175369
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col2"
                 class="data row9 col2" >
                    -0.173658
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col3"
                 class="data row9 col3" >
                    -0.0085519
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col4"
                 class="data row9 col4" >
                    -0.0495605
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col5"
                 class="data row9 col5" >
                    0.0704376
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col6"
                 class="data row9 col6" >
                    -0.121658
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col7"
                 class="data row9 col7" >
                    0.0603615
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col8"
                 class="data row9 col8" >
                    0.217381
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col9"
                 class="data row9 col9" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col10"
                 class="data row9 col10" >
                    -0.854982
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col11"
                 class="data row9 col11" >
                    -0.207392
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col12"
                 class="data row9 col12" >
                    -0.263935
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col13"
                 class="data row9 col13" >
                    -0.00306043
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col14"
                 class="data row9 col14" >
                    -0.0402023
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col15"
                 class="data row9 col15" >
                    -0.0511234
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col16"
                 class="data row9 col16" >
                    0.0889535
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col17"
                 class="data row9 col17" >
                    -0.0686996
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col18"
                 class="data row9 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row9_col19"
                 class="data row9 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row10" rowspan=1>
                    agency_name_Department of Public Works
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col0"
                 class="data row10 col0" >
                    0.046939
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col1"
                 class="data row10 col1" >
                    0.209533
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col2"
                 class="data row10 col2" >
                    0.206868
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col3"
                 class="data row10 col3" >
                    0.00880595
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col4"
                 class="data row10 col4" >
                    0.0629748
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col5"
                 class="data row10 col5" >
                    -0.0379802
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col6"
                 class="data row10 col6" >
                    0.102029
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col7"
                 class="data row10 col7" >
                    -0.0488367
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col8"
                 class="data row10 col8" >
                    -0.151716
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col9"
                 class="data row10 col9" >
                    -0.854982
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col10"
                 class="data row10 col10" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col11"
                 class="data row10 col11" >
                    -0.118411
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col12"
                 class="data row10 col12" >
                    -0.150695
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col13"
                 class="data row10 col13" >
                    -0.00174736
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col14"
                 class="data row10 col14" >
                    0.0393281
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col15"
                 class="data row10 col15" >
                    0.0262574
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col16"
                 class="data row10 col16" >
                    -0.0626754
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col17"
                 class="data row10 col17" >
                    0.0594136
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col18"
                 class="data row10 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row10_col19"
                 class="data row10 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row11" rowspan=1>
                    agency_name_Detroit Police Department
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col0"
                 class="data row11 col0" >
                    0.0386723
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col1"
                 class="data row11 col1" >
                    -0.0424905
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col2"
                 class="data row11 col2" >
                    -0.0418958
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col3"
                 class="data row11 col3" >
                    -0.000785474
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col4"
                 class="data row11 col4" >
                    -0.035747
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col5"
                 class="data row11 col5" >
                    -0.0665971
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col6"
                 class="data row11 col6" >
                    -0.0140086
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col7"
                 class="data row11 col7" >
                    -0.0484144
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col8"
                 class="data row11 col8" >
                    -0.0904623
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col9"
                 class="data row11 col9" >
                    -0.207392
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col10"
                 class="data row11 col10" >
                    -0.118411
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col11"
                 class="data row11 col11" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col12"
                 class="data row11 col12" >
                    -0.0365539
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col13"
                 class="data row11 col13" >
                    -0.000423856
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col14"
                 class="data row11 col14" >
                    0.0169206
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col15"
                 class="data row11 col15" >
                    0.0384459
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col16"
                 class="data row11 col16" >
                    -0.0428751
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col17"
                 class="data row11 col17" >
                    0.0154044
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col18"
                 class="data row11 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row11_col19"
                 class="data row11 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row12" rowspan=1>
                    agency_name_Health Department
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col0"
                 class="data row12 col0" >
                    -0.00555907
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col1"
                 class="data row12 col1" >
                    -0.0264583
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col2"
                 class="data row12 col2" >
                    -0.024933
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col3"
                 class="data row12 col3" >
                    0.000900588
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col4"
                 class="data row12 col4" >
                    0.00299099
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col5"
                 class="data row12 col5" >
                    -0.0277263
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col6"
                 class="data row12 col6" >
                    0.0680206
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col7"
                 class="data row12 col7" >
                    0.00634507
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col8"
                 class="data row12 col8" >
                    -0.0988538
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col9"
                 class="data row12 col9" >
                    -0.263935
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col10"
                 class="data row12 col10" >
                    -0.150695
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col11"
                 class="data row12 col11" >
                    -0.0365539
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col12"
                 class="data row12 col12" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col13"
                 class="data row12 col13" >
                    -0.000539416
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col14"
                 class="data row12 col14" >
                    -0.00753711
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col15"
                 class="data row12 col15" >
                    0.031004
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col16"
                 class="data row12 col16" >
                    -0.0343544
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col17"
                 class="data row12 col17" >
                    0.0155333
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col18"
                 class="data row12 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row12_col19"
                 class="data row12 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row13" rowspan=1>
                    agency_name_Neighborhood City Halls
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col0"
                 class="data row13 col0" >
                    -0.000699409
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col1"
                 class="data row13 col1" >
                    0.00237995
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col2"
                 class="data row13 col2" >
                    0.00245128
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col3"
                 class="data row13 col3" >
                    -0.00011423
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col4"
                 class="data row13 col4" >
                    -0.00393082
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col5"
                 class="data row13 col5" >
                    0.0017902
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col6"
                 class="data row13 col6" >
                    -0.00128837
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col7"
                 class="data row13 col7" >
                    0.00840098
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col8"
                 class="data row13 col8" >
                    -0.00235748
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col9"
                 class="data row13 col9" >
                    -0.00306043
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col10"
                 class="data row13 col10" >
                    -0.00174736
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col11"
                 class="data row13 col11" >
                    -0.000423856
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col12"
                 class="data row13 col12" >
                    -0.000539416
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col13"
                 class="data row13 col13" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col14"
                 class="data row13 col14" >
                    -8.73956e-05
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col15"
                 class="data row13 col15" >
                    -0.000765663
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col16"
                 class="data row13 col16" >
                    -0.00633805
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col17"
                 class="data row13 col17" >
                    0.011161
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col18"
                 class="data row13 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row13_col19"
                 class="data row13 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row14" rowspan=1>
                    disposition_Responsible (Fine Waived) by Deter
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col0"
                 class="data row14 col0" >
                    0.124956
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col1"
                 class="data row14 col1" >
                    -0.018466
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col2"
                 class="data row14 col2" >
                    -0.0173719
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col3"
                 class="data row14 col3" >
                    -0.0015961
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col4"
                 class="data row14 col4" >
                    0.00626365
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col5"
                 class="data row14 col5" >
                    -0.00736269
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col6"
                 class="data row14 col6" >
                    0.0202941
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col7"
                 class="data row14 col7" >
                    -0.000776949
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col8"
                 class="data row14 col8" >
                    -0.0153477
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col9"
                 class="data row14 col9" >
                    -0.0402023
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col10"
                 class="data row14 col10" >
                    0.0393281
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col11"
                 class="data row14 col11" >
                    0.0169206
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col12"
                 class="data row14 col12" >
                    -0.00753711
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col13"
                 class="data row14 col13" >
                    -8.73956e-05
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col14"
                 class="data row14 col14" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col15"
                 class="data row14 col15" >
                    -0.0106984
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col16"
                 class="data row14 col16" >
                    -0.0885597
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col17"
                 class="data row14 col17" >
                    -0.00783045
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col18"
                 class="data row14 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row14_col19"
                 class="data row14 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row15" rowspan=1>
                    disposition_Responsible by Admission
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col0"
                 class="data row15 col0" >
                    0.238997
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col1"
                 class="data row15 col1" >
                    -0.0640385
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col2"
                 class="data row15 col2" >
                    -0.145608
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col3"
                 class="data row15 col3" >
                    0.11791
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col4"
                 class="data row15 col4" >
                    -0.0273877
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col5"
                 class="data row15 col5" >
                    -0.0193673
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col6"
                 class="data row15 col6" >
                    -0.00194609
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col7"
                 class="data row15 col7" >
                    -0.0673838
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col8"
                 class="data row15 col8" >
                    -0.105844
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col9"
                 class="data row15 col9" >
                    -0.0511234
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col10"
                 class="data row15 col10" >
                    0.0262574
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col11"
                 class="data row15 col11" >
                    0.0384459
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col12"
                 class="data row15 col12" >
                    0.031004
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col13"
                 class="data row15 col13" >
                    -0.000765663
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col14"
                 class="data row15 col14" >
                    -0.0106984
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col15"
                 class="data row15 col15" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col16"
                 class="data row15 col16" >
                    -0.775862
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col17"
                 class="data row15 col17" >
                    -0.0686018
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col18"
                 class="data row15 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row15_col19"
                 class="data row15 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row16" rowspan=1>
                    disposition_Responsible by Default
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col0"
                 class="data row16 col0" >
                    -0.335455
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col1"
                 class="data row16 col1" >
                    0.0380344
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col2"
                 class="data row16 col2" >
                    0.111337
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col3"
                 class="data row16 col3" >
                    -0.0973869
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col4"
                 class="data row16 col4" >
                    0.0242223
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col5"
                 class="data row16 col5" >
                    0.0175426
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col6"
                 class="data row16 col6" >
                    -0.0256261
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col7"
                 class="data row16 col7" >
                    0.0497373
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col8"
                 class="data row16 col8" >
                    0.138091
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col9"
                 class="data row16 col9" >
                    0.0889535
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col10"
                 class="data row16 col10" >
                    -0.0626754
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col11"
                 class="data row16 col11" >
                    -0.0428751
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col12"
                 class="data row16 col12" >
                    -0.0343544
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col13"
                 class="data row16 col13" >
                    -0.00633805
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col14"
                 class="data row16 col14" >
                    -0.0885597
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col15"
                 class="data row16 col15" >
                    -0.775862
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col16"
                 class="data row16 col16" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col17"
                 class="data row16 col17" >
                    -0.567875
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col18"
                 class="data row16 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row16_col19"
                 class="data row16 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row17" rowspan=1>
                    disposition_Responsible by Determination
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col0"
                 class="data row17 col0" >
                    0.202819
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col1"
                 class="data row17 col1" >
                    0.0261695
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col2"
                 class="data row17 col2" >
                    0.0157017
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col3"
                 class="data row17 col3" >
                    0.00141495
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col4"
                 class="data row17 col4" >
                    -0.00385557
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col5"
                 class="data row17 col5" >
                    -0.00145961
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col6"
                 class="data row17 col6" >
                    0.0402406
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col7"
                 class="data row17 col7" >
                    0.0089375
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col8"
                 class="data row17 col8" >
                    -0.0796086
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col9"
                 class="data row17 col9" >
                    -0.0686996
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col10"
                 class="data row17 col10" >
                    0.0594136
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col11"
                 class="data row17 col11" >
                    0.0154044
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col12"
                 class="data row17 col12" >
                    0.0155333
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col13"
                 class="data row17 col13" >
                    0.011161
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col14"
                 class="data row17 col14" >
                    -0.00783045
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col15"
                 class="data row17 col15" >
                    -0.0686018
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col16"
                 class="data row17 col16" >
                    -0.567875
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col17"
                 class="data row17 col17" >
                    1
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col18"
                 class="data row17 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row17_col19"
                 class="data row17 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row18" rowspan=1>
                    disposition_Responsible - Compl/Adj by Default
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col0"
                 class="data row18 col0" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col1"
                 class="data row18 col1" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col2"
                 class="data row18 col2" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col3"
                 class="data row18 col3" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col4"
                 class="data row18 col4" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col5"
                 class="data row18 col5" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col6"
                 class="data row18 col6" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col7"
                 class="data row18 col7" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col8"
                 class="data row18 col8" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col9"
                 class="data row18 col9" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col10"
                 class="data row18 col10" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col11"
                 class="data row18 col11" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col12"
                 class="data row18 col12" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col13"
                 class="data row18 col13" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col14"
                 class="data row18 col14" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col15"
                 class="data row18 col15" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col16"
                 class="data row18 col16" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col17"
                 class="data row18 col17" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col18"
                 class="data row18 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row18_col19"
                 class="data row18 col19" >
                    nan
                
                
            </tr>
            
            <tr>
                
                
                <th id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085"
                 class="row_heading level0 row19" rowspan=1>
                    disposition_Responsible - Compl/Adj by Determi
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col0"
                 class="data row19 col0" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col1"
                 class="data row19 col1" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col2"
                 class="data row19 col2" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col3"
                 class="data row19 col3" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col4"
                 class="data row19 col4" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col5"
                 class="data row19 col5" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col6"
                 class="data row19 col6" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col7"
                 class="data row19 col7" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col8"
                 class="data row19 col8" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col9"
                 class="data row19 col9" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col10"
                 class="data row19 col10" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col11"
                 class="data row19 col11" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col12"
                 class="data row19 col12" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col13"
                 class="data row19 col13" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col14"
                 class="data row19 col14" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col15"
                 class="data row19 col15" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col16"
                 class="data row19 col16" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col17"
                 class="data row19 col17" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col18"
                 class="data row19 col18" >
                    nan
                
                
                
                <td id="T_80a9a6d2_88ae_11ec_b25d_c9f2c9305085row19_col19"
                 class="data row19 col19" >
                    nan
                
                
            </tr>
            
        </tbody>
        </table>
        



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
