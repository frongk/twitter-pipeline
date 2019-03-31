# twitter-pipeline
This model is supposed to help make it easier to develop Twitter applications that incorporate sentiment analysis. Running the model building script builds a `model` folder that you can easily import and use in your projects!

## About Sentiment
Sentiment is a score of English text that indicates how positive a particular message is. A Tweet like "You are the worst most terrible person ever!!!" will get a score close to 0. On the other hand, something really positive such as "this model is so wonderful! I love it! #omg" will get a score close to 1. 

Instead of building a model from scratch or having to prepocess text data to easily mesh with an existing Twitter sentiment model, you can simply enter an array of Tweets into this model and get a response.

Example Usage:
```python
In [1]: from model.twitter_pipeline import lgbm_model

In [2]: lgbm_model.predict_proba(['this model is so wonderful! i love it! #omg'])
Out[2]: array([[0.02152769, 0.97847231]])
```

This model has been trained and validated on a large set of Tweets (1,600,000) which have been labeled using emoticons as a proxy for the actual sentiment. The current form of the model is LightGBM which is gradient boosting model that uses trees. It is more efficient that XGBoost and should perform a little better when scoring Tweets.

The model performs pretty well. Training and testing ROC Areas Under the Curve are both 0.91. Here is the ROC curve:
![](https://raw.githubusercontent.com/frongk/twitter-pipeline/master/figures/roc_curve.png)

## Model Setup
Before the model can be used, it needs to be trained and serialized (pickled). To do this, simply run the `build_sentiment_pipeline_lgbm.py` file in the main directory.
```
$ python build_sentiment_pipeline_lgbm.py
```
This will create a data directory and download the raw dataset for training the model from the [Sentiment140 website](http://help.sentiment140.com/for-students). It will then build an scikit-learn pipeline object that takes raw texts and turns them into count vectors using word bagging. This feeds into the LightGBM.


The dependencies for the model build are:
`pandas`
`nltk`
`scikit-learn`
`lightgbm`

The analysis script also uses the `shap` package for generating force plots. 

## Usage
After building the model executing the `build_sentiment_pipeline_lgbm.py` script, copy the model directory into your project. You can import it using the line from above:
```python
from model.twitter_pipeline import lgbm_model
```
To actually predict the sentiment as a score (from 0 - negative to 1 - positive), use the line:
```python
lgbm_model.predict_proba(['this model is so wonderful! i love it! #omg'])[:,1]
```
The model produces two scores, a negative and a positive score. The index `[:,1]` pulls the positive score.

If you would like to just predict 1 for positive or 0 for negative and not have degree of emotion (i.e. how positive or how negative), then use:
```python
lgbm_model.predict(['this model is so wonderful! i love it! #omg'])
```

## How is it making decisions?
Here are some [SHAP force plots](https://github.com/slundberg/shap) from some of the Tweets from the dataset. This might provide a more intuitive sense of how the model is arriving at the final score. Note that a positive value is a more positive Tweet and a negative value is a more negative Tweet. Pink points to that word/feature making the score more positive. Blue points to that word/feature making the score more negative. 

Going to bed is exciting!!:
![](https://raw.githubusercontent.com/frongk/twitter-pipeline/master/figures/force_positive.PNG)

Seeing your old car is a touch sad but maybe bittersweet:
![](https://raw.githubusercontent.com/frongk/twitter-pipeline/master/figures/force_neutral.PNG)

This is a bummer:
![](https://raw.githubusercontent.com/frongk/twitter-pipeline/master/figures/force_negative.PNG)
