import pickle
from model.pipeline_util import TypeTransform

# usage:
# from model.twitter_pipeline import lgbm_model

lgbm_model = pickle.load(open('model/twitter_lgbm_800.pkl','rb'))
