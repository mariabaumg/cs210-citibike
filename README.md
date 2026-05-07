# **NYC Citi Bike Analysis**
## **Goal**
This project focuses on analyzing Citibike usage in New York City throughout the year. 
The goal was to find in what areas or what stations were being utilized the most and to cluster similarly behaved stations. 
Furthermore, the final goal was to create a model that shows predicted value of bike inflow at a specific station during specific season and day. Our ultimate output includes an interactive map that allows the user to select a station and view the predicted inflow of CitiBikes depending on season and day. The dataset that we utilized was from a credible source, Kaggle, that has multiple verifiable datasets. We utilized https://www.kaggle.com/datasets/leonczarlinski/citi-bike-nyc?select=202202-citibike-tripdata.csv

## **Execution**

To run any of the files and models in this repository make sure that the data processing parquet file which is run through data_prep.py in src is saved and accessible. The remaining files gather their data and information from this data parquet so it's important that the files are able to properly access it. 

To execute the Kmeans and Clustering data outputs, they can be run normally through the ipynb Notebook (km.ipynb). This file analyzes the stations and determines how to cluster them based on KMeans and the Elbow Method. The outputs can be found in the end with the graphs and each station mapped to its corresponding cluster. Each of the graphs helped analyze the behaviors of the different stations and how many clusters to create. 

For the model to run, since it is using large amounts of data, it took around 40 minutes to fully execute. To execute this model, first run through model_training.ipynb then move the processed json into the src folder, then either deploy website on gitpages or run "python -m http.server 8000" and go to http://localhost:8000 to run locally on the device. 

## **Authors**
Manasvini, Maggie, & Maria
