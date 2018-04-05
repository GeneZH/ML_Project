import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

interest_levels = ['low', 'medium', 'high']
tau = {
    'low': 0.69195995,
    'medium': 0.23108864,
    'high': 0.07695141,
    }
interest_levels = ['low', 'medium', 'high']

tau_train = {
    'low': 0.694683,
    'medium': 0.227529,
    'high': 0.077788,
    }

tau_test = {
    'low': 0.69195995,
    'medium': 0.23108864,
    'high': 0.07695141, }

def correct2(df, train=True, verbose=False):
    if train:
        tau = tau_train
    else:
        tau = tau_test

    df_sum = df[interest_levels].sum(axis=1)
    df_correct = df[interest_levels].copy()

    if verbose:
        y = df_correct.mean()
        a = [tau[k] / y[k]  for k in interest_levels]
        print( a)

    for c in interest_levels:
        df_correct[c] /= df_sum

    for i in range(20):
        for c in interest_levels:
            df_correct[c] *= tau[c] / df_correct[c].mean()

        df_sum = df_correct.sum(axis=1)

        for c in interest_levels:
            df_correct[c] /= df_sum

    if verbose:
        y = df_correct.mean()
        a = [tau[k] / y[k]  for k in interest_levels]
        print( a)

    return df_correct
def correct(df):
    y = df[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print (a)

    def f(p):
        for k in range(len(interest_levels)):
            p[k] *= a[k]
        return p / p.sum()

    df_correct = df.copy()
    df_correct[interest_levels] = df_correct[interest_levels].apply(f, axis=1)

    y = df_correct[interest_levels].mean()
    a = [tau[k] / y[k]  for k in interest_levels]
    print (a)

    return df_correct
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=86, num_rounds=1200):
        param = {}
        param['objective'] = 'multi:softprob'
        param['eta'] = 0.02
        param['max_depth'] = 6
        param['silent'] = 1
        param['num_class'] = 3
        param['eval_metric'] = "mlogloss"
        param['min_child_weight'] = 1
        param['subsample'] = 0.7
        param['colsample_bytree'] = 0.7
        param['seed'] = seed_val
        num_rounds = num_rounds

        plst = list(param.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
            xgtest = xgb.DMatrix(test_X, label=test_y)
            watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
            model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
        else:
            xgtest = xgb.DMatrix(test_X)
            model = xgb.train(plst, xgtrain, num_rounds)

        pred_test_y = model.predict(xgtest)
        return pred_test_y, model
if True:
    train_df = pd.read_json("./train.json")
    test_df = pd.read_json("./test.json")

    import re

    def cap_share(x):
        return sum(1 for c in x if c.isupper())/float(len(x)+1)

    for df in [train_df, test_df]:
        # do you think that users might feel annoyed BY A DESCRIPTION THAT IS SHOUTING AT THEM?
        df['num_cap_share'] = df['description'].apply(cap_share)

        # how long in lines the desc is?
        df['num_nr_of_lines'] = df['description'].apply(lambda x: x.count('<br /><br />'))

        # is the description redacted by the website?
        df['num_redacted'] = 0
        df['num_redacted'].ix[df['description'].str.contains('website_redacted')] = 1


        # can we contact someone via e-mail to ask for the details?
        df['num_email'] = 0
        df['num_email'].ix[df['description'].str.contains('@')] = 1

        #and... can we call them?

        reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
        def try_and_find_nr(description):
            if reg.match(description) is None:
                return 0
            return 1

        df['num_phone_nr'] = df['description'].apply(try_and_find_nr)

    import math
    def cart2rho(x, y):
        rho = np.sqrt(x**2 + y**2)
        return rho


    def cart2phi(x, y):
        phi = np.arctan2(y, x)
        return phi


    def rotation_x(row, alpha):
        x = row['latitude']
        y = row['longitude']
        return x*math.cos(alpha) + y*math.sin(alpha)


    def rotation_y(row, alpha):
        x = row['latitude']
        y = row['longitude']
        return y*math.cos(alpha) - x*math.sin(alpha)


    def add_rotation(degrees, df):
        namex = "rot" + str(degrees) + "_X"
        namey = "rot" + str(degrees) + "_Y"

        df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
        df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)

        return df

    def operate_on_coordinates(tr_df, te_df):
        for df in [tr_df, te_df]:
            #polar coordinates system
            df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
            df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
            #rotations
            for angle in [15,30,45,60]:
                df = add_rotation(angle, df)

        return tr_df, te_df

    train_df, test_df = operate_on_coordinates(train_df, test_df)

    image_date = pd.read_csv("./listing_image_time.csv")
    # rename columns so you can join tables later on
    image_date.columns = ["listing_id", "time_stamp"]

    # reassign the only one timestamp from April, all others from Oct/Nov
    image_date.loc[80240,"time_stamp"] = 1478129766

    image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
    image_date["img_date_month"]            = image_date["img_date"].dt.month
    image_date["img_date_week"]             = image_date["img_date"].dt.week
    image_date["img_date_day"]              = image_date["img_date"].dt.day
    image_date["img_date_dayofweek"]        = image_date["img_date"].dt.dayofweek
    image_date["img_date_dayofyear"]        = image_date["img_date"].dt.dayofyear
    image_date["img_date_hour"]             = image_date["img_date"].dt.hour
    image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

    test_df = pd.merge(test_df, image_date, on="listing_id", how="left")
    train_df = pd.merge(train_df, image_date, on="listing_id", how="left")

    # fix data
    test_df["bathrooms"].loc[19671] = 1.5
    test_df["bathrooms"].loc[22977] = 2.0
    test_df["bathrooms"].loc[63719] = 2.0
    test_df["bathrooms"].loc[17808] = 2.0
    test_df["bathrooms"].loc[22737] = 2.0
    test_df["bathrooms"].loc[837] = 2.0
    test_df["bedrooms"].loc[100211] = 5.0
    test_df["bedrooms"].loc[15504] = 4.0
    train_df["price"] = train_df["price"].clip(upper=30000)

    def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=626, num_rounds=1200):
        param = {}
        param['objective'] = 'multi:softprob'
        param['eta'] = 0.02
        param['max_depth'] = 6
        param['silent'] = 1
        param['num_class'] = 3
        param['eval_metric'] = "mlogloss"
        param['min_child_weight'] = 1
        param['subsample'] = 0.7
        param['colsample_bytree'] = 0.7
        param['seed'] = seed_val
        num_rounds = num_rounds

        plst = list(param.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
            xgtest = xgb.DMatrix(test_X, label=test_y)
            watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
            model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
        else:
            xgtest = xgb.DMatrix(test_X)
            model = xgb.train(plst, xgtrain, num_rounds)

        pred_test_y = model.predict(xgtest)
        return pred_test_y, model
    #test_df["bathrooms"].loc[19671] = 1.5
    #test_df["bathrooms"].loc[22977] = 2.0
    #test_df["bathrooms"].loc[63719] = 2.0
    #train_df["price"] = train_df["price"].clip(upper=31000)
    train_df["logprice"] = np.log(train_df["price"])
    test_df["logprice"] = np.log(test_df["price"])

    train_df["price_t"] =train_df["price"]/train_df["bedrooms"]
    test_df["price_t"] = test_df["price"]/test_df["bedrooms"]

    train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"]
    test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"]

    train_df['price_per_room'] = train_df['price']/train_df['room_sum']
    test_df['price_per_room'] = test_df['price']/test_df['room_sum']

    train_df["num_photos"] = train_df["photos"].apply(len)
    test_df["num_photos"] = test_df["photos"].apply(len)

    train_df["num_features"] = train_df["features"].apply(len)
    test_df["num_features"] = test_df["features"].apply(len)

    train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
    test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

    train_df["created"] = pd.to_datetime(train_df["created"])
    test_df["created"] = pd.to_datetime(test_df["created"])
    train_df["created_year"] = train_df["created"].dt.year
    test_df["created_year"] = test_df["created"].dt.year
    train_df["created_month"] = train_df["created"].dt.month
    test_df["created_month"] = test_df["created"].dt.month
    train_df["created_day"] = train_df["created"].dt.day
    test_df["created_day"] = test_df["created"].dt.day
    train_df["created_hour"] = train_df["created"].dt.hour
    test_df["created_hour"] = test_df["created"].dt.hour

    train_df["pos"] = train_df.longitude.round(3).astype(str) + '_' + train_df.latitude.round(3).astype(str)
    test_df["pos"] = test_df.longitude.round(3).astype(str) + '_' + test_df.latitude.round(3).astype(str)

    vals = train_df['pos'].value_counts()
    dvals = vals.to_dict()
    train_df["density"] = train_df['pos'].apply(lambda x: dvals.get(x, vals.min()))
    test_df["density"] = test_df['pos'].apply(lambda x: dvals.get(x, vals.min()))


    features_to_use=['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price','price_t',
    'price_per_room', 'logprice', 'density','num_photos', 'num_features', 'num_description_words',
    'listing_id', 'created_month', 'created_day', 'created_hour','img_date_month',
    'img_date_week','img_date_day','img_date_dayofweek','img_date_dayofyear','img_date_hour',
    'img_date_monthBeginMidEnd','img_days_passed', 'num_cap_share', 'num_nr_of_lines', 'num_redacted',
    'num_email', 'num_phone_nr', 'num_rho', 'num_phi', 'num_rot15_X',
    'num_rot15_Y', 'num_rot30_X', 'num_rot30_Y', 'num_rot45_X',
    'num_rot45_Y', 'num_rot60_X', 'num_rot60_Y']

    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(train_df)
    b=[np.nan]*len(train_df)
    c=[np.nan]*len(train_df)


    for i in range(5):
        building_level={}
        for j in train_df['manager_id'].values:
            building_level[j]=[0,0,0]

        test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index=list(set(index).difference(test_index))

        for j in train_index:
            temp=train_df.iloc[j]
            if temp['interest_level']=='low':
                building_level[temp['manager_id']][0]+=1
            if temp['interest_level']=='medium':
                building_level[temp['manager_id']][1]+=1
            if temp['interest_level']=='high':
                building_level[temp['manager_id']][2]+=1

        for j in test_index:
            temp=train_df.iloc[j]
            if sum(building_level[temp['manager_id']])!=0:
                a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
                b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
                c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])

    train_df['manager_level_low']=a
    train_df['manager_level_medium']=b
    train_df['manager_level_high']=c

    a=[]
    b=[]
    c=[]
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]

    for j in range(train_df.shape[0]):
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1

    for i in test_df['manager_id'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df['manager_level_low']=a
    test_df['manager_level_medium']=b
    test_df['manager_level_high']=c

    features_to_use.append('manager_level_low')
    features_to_use.append('manager_level_medium')
    features_to_use.append('manager_level_high')

    categorical = ['display_address', "manager_id", "building_id"]
    for f in categorical:
            if train_df[f].dtype=='object':
                lbl = LabelEncoder()
                lbl.fit(list(train_df[f].values) + list(test_df[f].values))
                train_df[f] = lbl.transform(list(train_df[f].values))
                test_df[f] = lbl.transform(list(test_df[f].values))
                features_to_use.append(f)





    train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

    train_df


    tfidf = CountVectorizer(stop_words='english', max_features=200)
    tr_sparse = tfidf.fit_transform(train_df["features"])
    te_sparse = tfidf.transform(test_df["features"])

    train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
    test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

    target_num_map = {'high':0, 'medium':1, 'low':2}
    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
    joblib.dump((train_X, train_y, test_X),"data.pkl",compress=3)
else:
    pass
print('ready')
preds, model = runXGB(train_X, train_y, test_X, num_rounds=1200)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("sub51.csv", index=False)

# l_time = pd.read_csv('pred6.csv')
# l_time2 = correct2(l_time)
# l_time2['listing_id']=l_time['listing_id']
# l_time2.to_csv("pred7.csv", index=False)
