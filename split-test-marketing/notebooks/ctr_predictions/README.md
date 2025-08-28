## CTR Predictions and Data Analysis
Can be found in the following notebook: [ctr_predictions.ipynb](./notebooks/ctr_predictions/ctr_predictions.ipynb) found in the notebooks folder
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