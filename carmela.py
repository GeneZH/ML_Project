import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
import collections
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.externals import joblib
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

class FeatureEngineering:

   def __init__(self, train, breakoutFeatures, yPresentInDataset):
      """
      ('headings are: ', [u'bathrooms', u'bedrooms', u'building_id', u'created', u'description', u'display_address', u'features', u'interest_level', u'latitude', u'listing_id', u'longitude', u'manager_id', u'photos', u'price', u'street_address'])
      """
      self.train = train.copy()

      self.preProcess(breakoutFeatures, yPresentInDataset)

      if ( yPresentInDataset ):
         self.y_train = self.train.interest_level.values
         self.train = self.train.drop(['interest_level'], axis=1)

      self.train = self.train.drop(['created', 'description', 'display_address', 'features', 'latitude', 'listing_id', 'longitude', 'photos', 'street_address'], axis=1)

      self.x_train = self.train.values
      
   def preProcess(self, breakoutFeatures = 0, yPresentInDataset=True):

   #- Make features column more uniform - "laundry room" and "laundry" are the same thing for example and all lower case
   #- Make display address lower case 
   #- Check for outliers/erroneous data and remove them
   #- Some building_ids are zero, need to delete these from the dataset

      # interest_level: Replace low, medium, high values with integer values 
      if ( yPresentInDataset ):
         y_map = {'low': 2, 'medium': 1, 'high': 0}
#         self.train['interest_level'] = self.train['interest_level'].apply(lambda x: y_map[x])

      # features: Make list of features lower case 
      self.train['features'] = self.train['features'].apply(lambda x: self.makeLowerCase(x))   

      # features: Break out sub-features (doorman, cats allowed, etc. into their own columns for most frequent features in the training set 
      if ( breakoutFeatures > 0 ):
         features = self.train['features'].values
         allFeatures = np.empty(1)
         featureCounts = {}
         orderedFeatureCounts = {}
         mostImportantFeatures = []
         numRowsFeatures = self.train['features'].size
         for i in range(0, numRowsFeatures):
            allFeatures = np.append(allFeatures, features[i])
            uniqueFeatures, uniqueIndices = np.unique(allFeatures, return_index=True)
            for j in range(0, allFeatures.size):

               # if feature not in dictionary at all, then add it
               if allFeatures[j] not in featureCounts:
                  featureCounts[allFeatures[j]] = 1

               # increment non-unique features
               if j not in uniqueIndices:
                  featureCounts[allFeatures[j]] += 1

            allFeatures = uniqueFeatures
  
#         print ("allFeatures are: ", allFeatures)
#         print ("allFeatures size is: ", allFeatures.size)
#         print ("featureCounts are: ", featureCounts)
       
         # add new column in train for most frequent sub-features (doorman, cats allowed, etc.) 
         orderedFeatureCounts = collections.OrderedDict(sorted(featureCounts.items(), key=lambda t: t[1], reverse=True))
#         print ("orderedFeatureCounts are: ", orderedFeatureCounts)
         mostImportantFeatures = itertools.islice(orderedFeatureCounts.items(), 0, breakoutFeatures)
         print ("Each of the below were very frequently mentioned as a feature in the 'features' column: ")
         for key, value in mostImportantFeatures:
            print ("mostImportantFeature in List: ", key, value)
            self.train[key] = self.train['features'].apply(lambda x: self.checkPresenceOfFeature(key, x))
         print ("\n")

         """ 
         for feat in featureCounts:
            if featureCounts[feat] > 5000 :
               mostImportantFeatures[feat] = featureCounts[feat]
#               print ("feat is: ", feat, featureCounts[feat])
         
         # add new column in train for most frequent sub-features (doorman, cats allowed, etc.)
         print ("Each of the below were very frequently mentioned as a feature in the 'features' column: ")
         for key, value in mostImportantFeatures:
            print ("most important feat is: ", key, value)
            self.train[key] = self.train['features'].apply(lambda x: self.checkPresenceOfFeature(key, x))
         print ("\n")
         """

      # created: Replace created with it's components year, month, etc.  
      self.train['Date'] = pd.to_datetime(self.train['created'])
      self.train['Year'] = self.train['Date'].dt.year
      self.train['Month'] = self.train['Date'].dt.month
      self.train['Day'] = self.train['Date'].dt.day
      self.train['Wday'] = self.train['Date'].dt.dayofweek
      self.train['Yday'] = self.train['Date'].dt.dayofyear
      self.train['hour'] = self.train['Date'].dt.hour
      self.train = self.train.drop(['Date'], axis=1)

      # latitude: Check for outliers and delete outliers 
#      print ("lats are: ", self.train['latitude'])
#      self.train['latitude'].apply(lambda x: self.checkLatsLongs(x, coord='lats'))

      # longitude: Check for outliers and delete outliers and make longitude values positive
#      print ("longs are: ", self.train['longitude'])
#      self.train['longitude'].apply(lambda x: self.checkLatsLongs(x, coord='longs'))
      self.train['longitude'] = self.train['longitude'].apply(lambda x: abs(x))

      # building_id: Use a map to replace building_ids with an integer value
      buildingIdMap = self.myStringMap('building_id')
      self.train = self.train[self.train.building_id != '0']
#      print ("self.train['building_id'] is: ", self.train['building_id'])
      self.train['building_id'] = self.train['building_id'].apply(lambda x: buildingIdMap[x]) 
   
      # display_address: Use a map to replace display_addresses with an integer value
      displayAddressMap = self.myStringMap('display_address')
      self.train['display_address'] = self.train['display_address'].apply(lambda x: displayAddressMap[x])

      # manager_id: Use a map to replace manager-ids with an integer value
      managerIdMap = self.myStringMap('manager_id')
      self.train['manager_id'] = self.train['manager_id'].apply(lambda x: managerIdMap[x])
     
      # photos_count: Add a new column for the number of photos for each listing
      self.train['photos_count'] = self.train['photos'].apply(lambda x: len(x))
      
   def makeLowerCase(self, x):
      for j in range(0, len(x)):
         x[j] = x[j].lower()
      return(x)

   def checkPresenceOfFeature(self, feat, x):
      if feat in x:
         return(1)
      else:
         return(0)
      
   def checkLatsLongs(self, x, coord='lats'):
      if ( coord is 'lats' and (x > 41 or x < 40) ):
         print("Value of Lat is: ", x)
      elif ( coord is 'longs' and (x < -74.1 or x > -73 ) ):
         print("Value of Long is: ", x)        

   def myStringMap(self, featureItem):
#      print ("featureItem is: ", featureItem)
      uniqueItems = self.train[featureItem].unique()
#      print ("num non-unique featureItem is: ", self.train[featureItem].size)
#      print ("num of unique featureItem is: ", uniqueItems.size)

      mapValues = np.arange(uniqueItems.size)
#      print ("mapValues are: ", mapValues)
      featureMap = dict( zip(uniqueItems, mapValues))
#      print ("featureMap is: ", featureMap)

      return (featureMap)

   def getTrainData(self):
      return(self.train)

   def univariateSelection(self, k=3):

      # feature extraction
      method = SelectKBest(score_func=chi2, k=3)
      fit = method.fit(self.x_train, self.y_train)
      # print column headers
#      headings = list(self.train.columns)
#      print("Feature set after pre-processin are: ", headings)
      # summarize scores
      np.set_printoptions(precision=3)
      uniVariateSelectionScores = fit.scores_

      ind = np.argpartition(uniVariateSelectionScores, -k)[-k:]
#      print("Scores are: ", fit.scores_)
#      print ("Indices are: ", ind)
      print ("Top Features using Univariant Selection are: ", self.train.columns[ind])

      features = fit.transform(self.x_train)
      # summarize selected features
#      print(features[0:5,:])

   def recursiveFeatureSelection(self, k=3):
      model = LogisticRegression()
      rfe = RFE(model, k)
      fit = rfe.fit(self.x_train, self.y_train)
      headings = list(self.train.columns)
#      print("headings are: ", headings)
#      print("Num Features: %d") % fit.n_features_
#      print("Selected Features: %s") % fit.support_
      ind = fit.get_support(indices=True)
#      print("Selected Features: %s") % ind
#      print("Feature Ranking: %s") % fit.ranking_
      print ("Top Features using Recursive Feature Selection are: ", self.train.columns[ind])
    
   def principleComponentAnalysis(self, k=3):
      pca = PCA(n_components=k)
      fit = pca.fit(self.x_train)
      # summarize components
#      print("Explained Variance: %s") % fit.explained_variance_ratio_
#      print(fit.components_) 

   def featureImportance(self, k=3):
      model = ExtraTreesClassifier()
      model.fit(self.x_train, self.y_train)
#      print(model.feature_importances_)
      ind = np.argpartition(model.feature_importances_, -k)[-k:]
      print ("Top Features using Feature Importance are: ", self.train.columns[ind])

if __name__ == "__main__":

   train_file = './train.json'
   test_file = './test.json'

   # read in the training and test data
   train = pd.read_json(train_file)
   test = pd.read_json(test_file)

   # select the number of features to pick using feature engineering
   pickNFeatures = 6

   # During pre-processing, breakout the 'features' column into the most frequently mentioned features (for example, "cats allowed", "no fee", "doorman", etc.)
   breakoutFeatures = 14

   print ("\n")
   print ("Number of features to select from training set is: ", pickNFeatures)
   print ("Number of sub-features to use from the 'features' column is: ", breakoutFeatures)
   print ("\n")

   featEng = FeatureEngineering(train, breakoutFeatures, True)
   features = list(train.columns)
   print("Train features before pre-processing are: ", features)
   print("\n")

   features = list(featEng.train.columns)
   print("Train features after pre-processing are: ", features)
   print("\n")

#   print ("Univariate Selection is: ")
   featEng.univariateSelection(k=pickNFeatures)
   print ("\n")

#   print ("Recursive Feature Selection is: ")
   featEng.recursiveFeatureSelection(k=pickNFeatures)
   print("\n")

#   print ("Principle Component Analysis is: ")
   featEng.principleComponentAnalysis(k=pickNFeatures)
#   print ("\n")

#   print ("Feature Importance is: ")
   featEng.featureImportance(k=pickNFeatures)

   # do pre-processing on the test data so the new test data looks like the modified training data
   featEng_test = FeatureEngineering(test, breakoutFeatures, False)   
   features_test = list(test.columns)
   print("Test features before pre-processing are: ", features_test)
   print("\n")

   features_test = list(featEng_test.train.columns)
   print("Test features after pre-processing are: ", features_test)
   print("\n")

   """
   kmeansFeatures = {'latitude' : train['latitude'],
                     'longitude' : train['longitude'],
#                     'interest_level' : train['interest_level']}
                     'price' : train['price']}

   kmeansFeaturesDF= pd.DataFrame(kmeansFeatures)

   print ("kmeansFeaturesDF are: ", kmeansFeaturesDF)
   kmeans = KMeans(n_clusters=3, random_state=0).fit(kmeansFeaturesDF)
   print ("kmeans.labels_ are: ", kmeans.labels_)
   print ("kmeans.cluster_centers_ are: ", kmeans.cluster_centers_)

   estimators = {'k_means_3': KMeans(n_clusters=3)}
#              'k_means_iris_8': KMeans(n_clusters=8),
#              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
#                                              init='random')}

   latitudes = train['latitude'].as_matrix()
   longitudes = train['longitude'].as_matrix()
   prices = train['price'].as_matrix()

   fignum = 1
   for name, est in estimators.items():
       fig = plt.figure(fignum, figsize=(4, 3))
       plt.clf()
       ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

       plt.cla()
       est.fit(kmeansFeaturesDF)
       labels = est.labels_

       ax.scatter(latitudes, longitudes, prices, c=labels.astype(np.float))

       ax.w_xaxis.set_ticklabels([])
       ax.w_yaxis.set_ticklabels([])
       ax.w_zaxis.set_ticklabels([])
       ax.set_xlabel('Latitude')
       ax.set_ylabel('Longitude')
       ax.set_zlabel('Price')
       fignum = fignum + 1
       plt.show()
   """
   """
   categoricals = [x for x in train_test.columns if train_test[x].dtype == 'object']

   for feat in categoricals:
       lbl = preprocessing.LabelEncoder()
       lbl.fit(list(train_test[feat].values))
       train_test[feat] = lbl.transform(list(train_test[feat].values))

   features = list(train_test.columns)
 
#   print("features are: ", features)
   """
