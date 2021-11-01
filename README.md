# Replica Analytics - Questions MLE
 
## Questions 1-3
Questions 1-3 have been answered throughout the notebook. You can view the notebook within this repository or at [nbviewer.com](https://nbviewer.org/github/nefrank/replica-analytics-questions-mle/blob/main/USCensus.ipynb?flush_cache=True) for better readability.

## Question 4
A Flask REST app named '/API/app.py' can be run to demonstrate the API. It can train and predict for both models based on the uploaded US Census data. Model parameters can be selected and models can be named and saved. Saved models can be selected to make a prediction based on a fillable form containing the features from the dataset with the result displayed.

Here is a walkthrough of the process:

Upload US Census data:
![image](https://user-images.githubusercontent.com/72168799/139619237-fe3a84ad-f8d2-44fc-96ac-a8411e32cd40.png)

Select model parameters + name and train model:
![image](https://user-images.githubusercontent.com/72168799/139619415-8b7e83a9-b78e-4cac-abd4-09802b67c8ae.png)

View training results for the model. Using the selected model, predict based on the fillable form:
![image](https://user-images.githubusercontent.com/72168799/139619489-b8b6680e-b41f-45e9-b2cd-fa8149547093.png)

Predicted income is returned:
![image](https://user-images.githubusercontent.com/72168799/139619521-960e2620-8838-4b20-a0ed-5908daa110b3.png)

## Question 5
A unit testing file named '/API/test.py' can be run to perform the unit tests on the web app and API. The tests are described within the comments of the file.

## (Bonus) Question 6
I used the 'waitress' package to deploy my flask app with an adjustable thread count, I then used the 'locust' package to simulate requests to the API. I chose to simulate training requests as they take a few seconds. 

The reports can be seen here:
 - [Report: 2 Cores](https://htmlpreview.github.io/?https://github.com/nefrank/replica-analytics-questions-mle/blob/main/reports/report_2cores.html)
 - [Report: 4 Cores](https://htmlpreview.github.io/?https://github.com/nefrank/replica-analytics-questions-mle/blob/main/reports/report_4cores.html)

When the API is limited to 2 cores, it could handle 1 and 2 users at the same rate with a wait time of around 4s. When the users increased to 4, the time doubles to around 8s because there are only 2 cores to use. The same trend occurs as the time doubles to around 17s when 8 users are invoking the API. At 20 users, the time increases up to 43s. This result is expected as there are 10x as many users as cores, therefore each user must wait 10x the initial waiting time.

When the number of cores is increased to 4 the wait time was around 4-5s for up to 4 users, as expected when each user gets their own processing. When the number of users doubles to 8, the wait times also double to around 8-9s. At 20 users, the wait time is around 25s which again is 5x the initial wait time since there are 5x as many users as cores.
