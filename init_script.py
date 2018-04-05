import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, model_selection
import string
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.externals import joblib
from scipy import sparse

train_file = './train.json'
test_file = './test.json'



train = pd.read_json(train_file)
test = pd.read_json(test_file)
listing_id = test.listing_id.values


y_map = {'low': 2, 'medium': 1, 'high': 0}
train['interest_level'] = train['interest_level'].apply(lambda x: y_map[x])
y_train = train.interest_level.values

train = train.drop(['listing_id', 'interest_level'], axis=1)
test = test.drop('listing_id', axis=1)

ntrain = train.shape[0]

train_test = pd.concat((train, test), axis=0).reset_index(drop=True)

train_test['Date'] = pd.to_datetime(train_test['created'])
train_test['Year'] = train_test['Date'].dt.year
train_test['Month'] = train_test['Date'].dt.month
train_test['Day'] = train_test['Date'].dt.day
train_test['Wday'] = train_test['Date'].dt.dayofweek
train_test['Yday'] = train_test['Date'].dt.dayofyear
train_test['hour'] = train_test['Date'].dt.hour

train_test = train_test.drop(['Date', 'created'], axis=1)


train_test['features_count'] = train_test['features'].apply(lambda x: len(x))
train_test['features2'] = train_test['features']
train_test['features2'] = train_test['features2'].apply(lambda x: ' '.join(x))

c_vect = CountVectorizer(stop_words='english', max_features=200, ngram_range=(1, 1))
c_vect.fit(train_test['features2'])

c_vect_sparse_1 = c_vect.transform(train_test['features2'])
c_vect_sparse1_cols = c_vect.get_feature_names()



train_test.drop(['features', 'features2'], axis=1, inplace=True)
train_test.drop(['description'], axis=1, inplace=True)


train_test['photos_count'] = train_test['photos'].apply(lambda x: len(x))
train_test.drop(['photos', 'display_address', 'street_address'], axis=1, inplace=True)

categoricals = [x for x in train_test.columns if train_test[x].dtype == 'object']

for feat in categoricals:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_test[feat].values))
    train_test[feat] = lbl.transform(list(train_test[feat].values))

features = list(train_test.columns)


train_test_cv1_sparse = sparse.hstack((train_test, c_vect_sparse_1)).tocsr()


x_train = train_test_cv1_sparse[:ntrain, :]
x_test = train_test_cv1_sparse[ntrain:, :]
features += c_vect_sparse1_cols

SEED = 77
NFOLDS = 5

params = {
    'eta':.01,
    'colsample_bytree':.8,
    'subsample':.8,
    'seed':77,
    'nthread':16,
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':3,
    'silent':1
}


dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test)

bst = xgb.cv(params, dtrain, 10000, NFOLDS, early_stopping_rounds=50, verbose_eval=25)

best_rounds = np.argmin(bst['test-mlogloss-mean'])

bst = xgb.train(params, dtrain, best_rounds)

preds = bst.predict(dtest)

preds = pd.DataFrame(preds)

cols = ['high', 'medium', 'low']

preds.columns = cols

preds['listing_id'] = listing_id

preds.to_csv('my_preds.csv', index=None)

joblib.dump([dtrain,dtest,bst],'data.pkl',compress=3)
