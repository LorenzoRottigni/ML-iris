from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# static matrix
X = [
    [110, 1.70, 'rugby'],
    [100, 1.90, 'basket'],
    [120, 1.90, 'rugby'],
    [70, 1.60, 'soccer'],
]

transformers = [
    [
        # Name of transformer
        'category_vectorizer',
        # Alrorithm of transformation
        OneHotEncoder(),
        # Columns to be transformed, in this case the sport category column
        [2]
    ],
    # This might be a long list of transformers
    # able to perform complex operations on source dataset
]

ct = ColumnTransformer(
    # Transformation config
    transformers,
    # Normally it would drop the unused columns
    # this allows the transform on specified column and keep the others as are 
    remainder='passthrough'
)

ct.fit(X)

X = ct.transform()

# Short hand
#X = ct.fit_transform(X)