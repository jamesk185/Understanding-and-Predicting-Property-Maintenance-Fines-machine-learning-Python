
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


```python

```
