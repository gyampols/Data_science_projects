# Data Science Portfolio - Gregory Yampolsky

Hi and welcome to my GitHub portfolio for personal data science projects. Each project has its own folder and README with reproducible steps.

- **Email**: [gregyampolsky@gmail.com](gregyampolsky@gmail.com)
- **LinkedIn**: [linkedin.com/gregory-yampolsky](https://www.linkedin.com/in/gregory-yampolsky-042159172/)

## Achievements
- [Publication](https://arxiv.org/abs/2408.06679): Case-based Explainability for Random Forest: Prototypes, Critics, Counter-factuals and Semi-factuals.
- Graduate of Stevens Institute of Technology with distinction (3.91 GPA).

## Projects
<img align="left" width="250" height="150" src=./Images/a_bsplittesting.jpg> **[SplitTestMarketing](./SplitTestMarketing)**

Skills: Classification, A/B Testing, Clustering

 A website that offers online tools to connect home-buyers with mortgage lenders has done a split test on different call to action button copies and placements.  The goal of this project was two-fold.  1. predict the probability of a user clicking the call to action button. 2. Determine the ideal copy and placement combination to maximize revenue.

 <img align="left" width="250" height="150" src=./Images/rent-estimate.png> **[Rent Price Prediction](https://github.com/gyampols/RentEstimateProject/tree/main)**

Skills: Regression, shapley values, feature importance, web scraping, Geospatial data

The company wants a Python model that estimates market rent for single-family homes across the U.S. using basic property info and location. They gave two files: one to train on (with past rented prices over the last two years) and one to test on. Train a model on the provided historical data (TrainingSet.csv) using: Latitude, Longitude Bedrooms, Bathrooms Square Feet Year Built Plus the target: Close Price (i.e. the rent it actually leased for) Use that model to predict a “Market Rent” for new properties (like the ones in TestSet.csv or any similar dataframe). Deliver a function that takes a pandas DataFrame with those columns and returns the same DF with an extra Market Rent column. 

 <img align="left" width="200" height="200" src=./Images/FitnessScheduleIcon.png> **[Ai Activity Planner](https://github.com/gyampols/Ai-Activity-Planner/tree/main)**

Skills: html, java, api calls, Flask, GCP, SQL, Docker

AI Activity Planner helps you maintain a balanced and active lifestyle by intelligently scheduling your favorite activities throughout the week. Whether you're into running, yoga, swimming, or any other activity, our AI-powered system creates personalized plans that consider your preferences, fitness levels, and schedule constraints.

 ## Mini Projects
- - **[basic-regression](./basic-regression)**

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
