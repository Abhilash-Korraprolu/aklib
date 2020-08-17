
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    import pandas as pd
    import numpy as np
    import scipy.stats as ss

except ImportError:
    raise ImportError(
        "Packages: sklearn, pandas, numpy, scripy are required to be installed") from None


# Custom sklearn transformer to reduce skewedness of features using Box-Cox Transformations

class unskew(BaseEstimator, TransformerMixin):
    '''
    Features:
    1. Accepts non-positive inputs
    2. Skips features where the transformation hasn't reduced skewdness.
    3. Can be fitted with a 2D array or a dataframe.
    4. Can be used individually & in sklearn pipelines.

    To do: Incorporate Tukey's Ladder.
    '''

    def toArr(self, X):
        return X.to_numpy() if isinstance(X, pd.DataFrame) else X

    def boxCoxTransform(self, X, col, lmbda=None):
        c = 0 if min(X[:, col]) > 0 else 1 - min(X[:, col])

        return ss.boxcox(X[:, col] + c, lmbda=lmbda)

    def getLambda(self, X, col):

        X = self.toArr(X)

        # boxcox returns [0]- transformed, [1]- lambda value
        bc = self.boxCoxTransform(X, col)
        lmbda = bc[1]

        # we only need the magnitude of the skew to compare
        originalSkew = abs(ss.skew(X[:, col]))
        bcSkew = abs(ss.skew(bc[0]))

        # return lambda if boxcox reduces the skew. Else return nan to indicate to not transform.
        return lmbda if bcSkew < originalSkew else np.nan

    def fit(self, X, y=None):

        X = self.toArr(X)

        lmbdas = []
        for i in range(X.shape[1]):
            lmbdas.append(self.getLambda(X, i))

        self.lambdas_ = lmbdas

        return self

    def transform(self, X, y=None):

        X = self.toArr(X)

        for i in range(X.shape[1]):

            # transform only if the skew is reducing. Nan indicates that the skew increased after boxcox.
            if ~np.isnan(self.lambdas_[i]):
                X[:, i] = self.boxCoxTransform(X, i, lmbda=self.lambdas_[i])

        return X

    def fit_transform(self, X, y=None):

        self = self.fit(X)
        return self.transform(X, y)


def getSkew(X, display_skewed_features=False):

    skewed_cols = 0
    num_cols = X.select_dtypes(include='number').columns
    for col in num_cols:

        skew = X[col].skew()
        if abs(skew) > 1:

            if display_skewed_features = True:
                print(col, ':', skew)

            skewed_cols += 1

    print('\n', skewed_cols, 'features are significantly skewed.')


class dummy(BaseEstimator, TransformerMixin):

    '''
    dummy class: Done nothing. Handy during pipelines.
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        self = self.fit(X, y)
        return self.transform(X, y)
