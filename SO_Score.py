import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from sklearn.feature_selection import chi2
import nltk
import string
import re
import math
from nltk.corpus import stopwords


# Positive and negative vocabularies as seed to calculate the pointwise mantual information
# Combined with different versions, feel free to amend them
positive_vocab = [
    'good', 'nice', 'great', 'awesome', 'outstanding',
    'fantastic', 'terrific', 'like', 'love', 'fortunately', 'excellent']
negative_vocab = [
    'bad', 'terrible', 'crap', 'useless', 'hate', 'poor', 'wrong', 'unfortunately',
    'dissappointed',   'expensive']

# save the scores
def save_txt(score_list,filename, mode='a'):
    file = open(filename, mode)
    for i in range(len(score_list)):
        file.write(str(score_list[i])+'\n')
    file.close()


class buildDic:

    def __init__(self, train):
        """
        ('headings are: ', [u'bathrooms', u'bedrooms', u'building_id', u'created', u'description', u'display_address', u'features', u'interest_level', u'latitude', u'listing_id', u'longitude', u'manager_id', u'photos', u'price', u'street_address'])
        """
        self.train = train.copy()

        self.documents = self.train['description']
        self.documents = list(self.documents)
        self.n_docs = len(self.documents)

        # Preprocessing
        new_doc = []
        for description in self.documents:
            # Replace all punctuations and numbers with spaces
            regex = re.compile('[%s]' % re.escape(string.punctuation+"123456789"))
            description = description.lower()
            out = regex.sub(' ', description)
            new_doc.append(out)

        self.documents = new_doc

        u_dic, b_dic = self.build_dic(self.documents)
        print(b_dic)
        p_u, p_b = self.get_probobilities(u_dic, b_dic, self.n_docs)
        print(p_b)
        so = self.get_pmi(p_u, p_b)
        scores = self.calculate_score(so, self.documents)
        print("lines of records:")
        print(len(scores))
        save_txt(scores, 'score.txt')



    # to build unigram freq dic (it means the freq of a word appearing in a document, duplicate ones do not count)
    # and bigram freq dic (it means the freq of 2 words that appear in the same document, duplicate ones will be counted)
    def build_dic(self, documents):
        u_dic = defaultdict(int)
        b_dic = defaultdict(int)

        for description in documents:

            tem_udic = defaultdict(int)

            filtered_words = [word for word in description.split() if word not in stopwords.words('english')]
            for word in filtered_words:
                tem_udic[word] += 1
            for key in tem_udic.keys():
                u_dic[key] += 1
            # new dic for each piece of description
            tem_udic.clear()

        # Remove the words with lowest freq (1) from dictionary
        u_dic_cp = u_dic.copy()
        for key, value in u_dic_cp.items():
            if value == 1:
                del u_dic[key]

        for description in documents:
            for word in description.split():
                tem_udic[word] += 1
            for key in tem_udic.keys():
                u_dic[key] += 1
                for second_key in tem_udic.keys():
                    if key != second_key:
                        b_dic[(key, second_key)] += 1
            # new dic for each piece of description
            tem_udic.clear()

        return u_dic, b_dic

    def get_probobilities(self, u_dic, b_dic, n_docs):

        p_u = defaultdict(float)
        p_b = defaultdict(float)

        for term, n in u_dic.items():
            p_u[term] = n / n_docs
            for key in list(b_dic.keys()):
                term2 = key[1]
                if term != term2:
                    p_b[(term, term2)] = b_dic[(term, term2)] / n_docs

        return p_u, p_b

    # Algorithm of PMI
    def get_pmi(self, p_u, p_b):
        pmi = defaultdict(float)
        for t1 in p_u:
            for t2 in p_u:
                if t1 != t2:
                    m = p_u[t1] * p_u[t2]
                    pmi[(t1, t2)] = math.log2(p_b[(t1, t2)] / m)

        semantic_orientation = {}
        for term, n in p_u.items():
            positive_assoc = sum(pmi[(term, tx)] for tx in positive_vocab)
            negative_assoc = sum(pmi[(term, tx)] for tx in negative_vocab)

            #if the sum is larger then 0, positive; 0, neutral; otherwise, negative
            semantic_orientation[term] = positive_assoc - negative_assoc

        return semantic_orientation


    # Calculate the sentiment orientation score by summing up every words scores
    def calculate_score(self, so, documents):
        score_list = []
        for description in documents:
            score = 0.0
            for word in description.split():
                score += so[word]
            score_list.append(score)
        return score_list

if __name__ == "__main__":

    train_file = './train.json'
    test_file = './test.json'

    # read in the training and test data
    train = pd.read_json(train_file)
    test = pd.read_json(test_file)

    buildDic(train)

