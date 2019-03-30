# for getting raw data
import os
import requests
import zipfile
from tqdm import tqdm

# for training model and building pipeline
import pandas
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline

from model.pipeline_util import TypeTransform


import pickle

N_ESTIMATORS = 800

# make dir to store data
if not os.path.exists('data'):
    print('making data dir')
    os.mkdir('data')

# download data file
if not os.path.exists('data/raw_data.zip'):
    print('getting data file from sentiment140')
    data_url = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
    response = requests.get(data_url, stream=False)

    with open('data/raw_data.zip', 'wb') as handle:
        handle.write(response.content)

    # response = requests.get(data_url, stream=True)
    # with open('data/raw_data.zip', 'wb') as handle:
    #     for data in tqdm(response.iter_content(), desc='downloading/writing file'):
    #         handle.write(data)

# unzip data
if not os.path.exists('data/training.1600000.processed.noemoticon.csv'):
    print('unzipping data file')
    zip_ref = zipfile.ZipFile('data/raw_data.zip', 'r')
    zip_ref.extractall('data/')
    zip_ref.close()


# pull and process data
print('pulling data and generating training and validation sets')
data = pandas.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1',header=None)  #.sample(10000)
raw_x = data[5]
raw_y = data[0].map(lambda x: 1 if x==4 else 0)

X_train, X_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.3)

# items
print('building pipeline object')
tweet_tokenizer = TweetTokenizer(strip_handles=True)
c_vect = CountVectorizer(tokenizer=tweet_tokenizer.tokenize, 
                         ngram_range=(1,3),
                         lowercase=True,
                         max_df=0.2, 
                         max_features=40000)


t_trans=TypeTransform(astype='float64')

clf = lightgbm.LGBMClassifier(num_leaves=31,
                             learning_rate=0.3,
                             n_estimators=N_ESTIMATORS,
                             n_jobs=3,
                             silent=False)

estimator = Pipeline(memory=None,
                     steps=[('countvect', c_vect), 
                            ('typetrans', t_trans),
                            ('lgbmmodel', clf)])

print('fitting pipeline object')
estimator.fit(X_train, y_train)#,
              #lgbmmodel__eval_set=[(X_test, y_test)],
              #lgbmmodel__eval_metric='auc',
              #lgbmmodel__early_stopping_rounds=5)

y_pred = estimator.predict_proba(X_test)[:,1]
auc_score = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc_score}')

pickle.dump(estimator, open(f'model/twitter_lgbm_{N_ESTIMATORS}.pkl','wb'))
