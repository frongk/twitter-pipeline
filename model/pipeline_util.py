from sklearn.base import TransformerMixin

class TypeTransform(TransformerMixin):
    def __init__(self, astype=None):
        super().__init__()
        self.astype=astype

    def fit(self, X, y=None, **fit_parmas):
        return self

    def transform(self, X, **transform_params):
        return X.astype(self.astype)

    def get_params(self, deep=True):
        return dict()
