# twitter-pipeline
This model is supposed to help make it easier to develop Twitter applications that incorporate sentiment analysis. Instead of building a model from scratch or having to prepocess text data to easily mesh with an existing Twitter sentiment model, you can simply enter an array of Tweets into this model and get a response.

Example Usage:
```python

```

This model has been trained and validated on a large set of Tweets (1,600,000) which have been labeled using emoticons as a proxy for the actual sentiment. The current form of the model is LightGBM which is gradient boosting model that uses trees. It is more efficient that XGBoost and should perform a little better when scoring Tweets.

The model performs pretty well. Training and testing ROC Areas Under the Curve are both 0.91. Here is the ROC curve:
![]()

## Model Setup
Before the model can be used, it needs to be trained and serialized (pickled). To do this, simply run the `build_sentiment_pipeline_lgbm.py` file in the main directory.
```
$ python build_sentiment_pipeline_lgbm.py
```
This will create a data directory and download the raw dataset for training the model from the [Sentiment140 website](http://help.sentiment140.com/for-students). It will then build an scikit-learn pipeline object that takes raw texts and turns them into count vectors using word bagging. 


The dependencies for this file are:
