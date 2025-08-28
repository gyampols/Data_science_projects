
# split-test-marketing

<img align="center" width="250" height="150" src=./Images/a-b-split.png>

The goal of this project was two-fold.

1. **[CTR Predictions](./notebooks/ctr_predictions/)**: predict the probability of a user clicking the call to action button.

2. **[Segmentation and hypothesis testing](./notebooks/segmentation_hypothesis_testing/)**: Determine the ideal (ctaCopy x ctaPlacement) combination to maximize revenue.

You can find the notebooks for each task in the [notebooks folder](./notebooks). 

### Description
Financial Services stands out as the premier financial authority. When you visit Financialservices.com, the reviews, guides, and educational content have been developed by leading personal finance experts. Financial Services’s product comparison tools, calculators, and educational content help over 100 million consumers make smarter financial decisions each year. No matter where you are in your financial journey, Financial Services can help you reach your goals. 
One particular area in which Financial Services places a strong emphasis is mortgages, with the goal of advancing the visitor's decision-making journey and ultimately guiding them toward applying for and securing a mortgage. The mortgage team’s goal is to **maximize revenue** generated from users on our website by getting them to **schedule an appointment** with one of Financial Services’ mortgage partners. Once the appointment is scheduled, Financial Services gets paid a bounty (revenue). The amount depends on the variation of mortgage chosen for the appointment. For the purpose of this case, there are 4 variations (A, B, C, D), each with their own bounty associated with them. 

The customer journey is straightforward. A user visits a page on the website. Once on the page, they have the option to click on a banner that takes them to a form that they can fill out to schedule an appointment with a mortgage lender. They have the option to choose from four mortgage types when they schedule the appointment.

To support this goal, the mortgage team conducted a split test, also called A/B testing, on various mortgage pages. This testing allows the team to compare the performance of website creative variations to see which one appeals more to visitors to maximize a targeted metric. 
There were two features that the team tested.

**CTA Copy**

<img align="center" width="300" height="250" src=./Images/ctaCopy.PNG>

**CTA Placement**

<img align="center" width="250" height="150" src=./Images/ctaPlacement.PNG>


The objective of this test was to learn which banner’s call-to-action (CTA) title copy and on-page placement combination will best entice visitors to click and enter the scheduling form in hope of increasing appointments and overall revenue. 
### Data
The testing has concluded and we have a data set of 120,000 decisions (rows) of the combination of CTA Copy and CTA placement being served randomly to users coming to the website. To help increase revenue per decision, the team believes that certain groups of people may have performed differently based on the CTA copy and placement. We can use an algorithm to predict in real-time what CTA combination would be best to show to a future user to maximize the key metric.

For assessment purposes, these data have been partitioned into 2 sets:
train.csv
test.csv

Train.csv includes 100,000 rows of the original data with labels, test.csv contains 20,000 rows of the original data without labels. You will use test.csv to make predictions for submission. 


| Variables | Definition |
| :---- | :---- |
| userId | Unique identifier for users visiting Financial Services |
| sessionReferrer | Source from which the user arrived (e.g., search engine) |
| browser | Browser used by the user (e.g., Chrome, Firefox) |
| deviceType | Category of device used by the user (e.g., mobile, desktop, tablet) |
| estimatedAnnualIncome | Estimated annual income of the user based on geographic location |
| estimatedPropertyType | Estimated property type (e.g., residential, commercial) |
| visitCount | Number of previous visits did the user have before the current visit |
| pageURL | URL of the current page visited |
| ctaCopy | Text of the call to action that was prompted to the user for potential engagement |
| ctaPlacement | Location of the call-to-action button on the page |
| editorialSnippet | Text surrounding the call-to-action |
| scrollDepth | Percentage of the page length scrolled by the user.   If the user didn’t scroll on the page AFTER the ctaCopy and ctaPlacement already loaded, than scrollDepth will be 0\. |
| clickedCTA | If the user clicked the call-to-action (yes 1/no 0\) |
| submittedForm | If a user submitted the mortgage application form (yes 1/no 0\) |
| scheduledAppointment | If the user scheduled an appointment regarding the mortgage application (yes 1/no 0\) |
| mortgageVariation | Mortgage product variation the user applied for to discuss during the appointment. This only is present when someone scheduled an appointment |
| revenue | Revenue Financial Services received from the appointment scheduling and mortgage variation. If no scheduled appointment, this will be 0\. |


## [CTR Predictions and Data Analysis](./notebooks/ctr_predictions/)
<img align="center" width="250" height="150" src=./Images/Examples-of-a-Call-to-Action.webp>

Can be found in the following notebook: [ctr_predictions](./notebooks/ctr_predictions/ctr_predictions.ipynb)
#### Part 1 Predictive Model Building

Built a predictive model to make predictions on an unlabeled test set test.csv. The predictions are the probability of **ClickedCTA** Pr(ClickedCTA \= 1\) in \[0, 1\].  Evaluation metric given by the stakeholders is to minimize log loss.

#### Part 2 Answering Questions

Provided answers to several questions: 

1. *What relevant key metrics are provided to evaluate the CTA combinations? And which CTA Copy and CTA Placement did best/worst based on the key metrics?*   
2. *Which groups of people tend to be more correlated or less correlated with our key metrics?*  
3. *What ways can you manipulate the columns/dataset to create features that increase predictive power towards our key metric?*  
4. *Besides Log Loss, what other metrics will you use to evaluate the model's performance, and why?*

#### Extra: 

A few extra questions that were answered: 

1. *What additional predictive model would you build to inform which CTA combinations would maximize the revenue in addition to predict ClickedCTA, and why?*   
2. *If we called one of these CTA combinations our champion (serve it 100% of the time), how much incrementally is that worth to us vs. the average of the rest of the split test?* 



## [User Segmentation and Hypothesis Testing](./notebooks/segmentation_hypothesis_testing/)
<img align="center" width="250" height="150" src=./Images/Market-Segmentation.png>

Can be found in the following notebook: [ctr_predictions](./notebooks/segmentation_hypothesis_testing/segmentation_hypothesis_testing.ipynb)

Determine the ideal (ctaCopy x ctaPlacement) combination to maximize revenue.

## CLI

```bash
# Train
cd "split-test-marketing"
python -m src.train --train_csv data/raw/train.csv --out_model models/model.joblib --target target --id_col userId

# Predict
python -m src.predict --model models/model.joblib --features_csv data/raw/test.csv --out_csv outputs/predictions/predictions.csv --id_col userId

# Evaluate
python -m src.evaluate --preds_csv outputs/predictions/predictions.csv --truth_csv data/raw/train.csv --id_col userId --target target
```
