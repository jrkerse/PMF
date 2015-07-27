from __future__ import division

import numpy as np
from scipy import sparse
import sys


class PMF:

    k = 25            # used for pseudo averages
    K = 0.015         # regularization parameter
    min_iter = 100    # minimum number of iterations
    min_imp = 1e-4    # improvement threshold


    def __init__(self, file_path, rank=40, init=0.1):
        
        self.rank=rank
        
        self.sparse_matrix = self._load_data(file_path)
        matrix = self.sparse_matrix.todense()
        
        self.mu, user_bias, item_bias = self.baseline(matrix)
        self.user_matrix = self._create_matrix(user_bias, rank, init)
        self.item_matrix = self._create_matrix(item_bias, rank, init)
        
#         print 'user: {userbias}'.format(userbias=user_bias.shape)
#         print 'item: {itembias}'.format(itembias=item_bias.shape)
        
        self.sparse_matrix = self._load_data(file_path)
        users, items = self.sparse_matrix.nonzero()
        self.user_pseudo_rating = dict.fromkeys(users)
        self.item_pseudo_rating = dict.fromkeys(items)
        self._compute_pseudo_averages()


    def baseline(self, matrix):
        '''
        Returns the global average rating, user bias, and item bias.
        '''
        mu = self._get_mu(matrix)
        user_bias = mu - [self._get_mu(matrix[i,:]) for i in xrange((matrix.shape[0]))]
        item_bias = mu - [self._get_mu(matrix[:,i]) for i in xrange((matrix.shape[1]))]
        return mu, user_bias, item_bias


    def _get_mu(self, x):
        return np.nansum(x) / np.count_nonzero(~np.isnan(x))


    def _create_matrix(self, bias, rank, init):
        return np.ones((rank, bias.shape[0])) * init


    def _get_user_rating(self, user_id):
        user_reviews = self.sparse_matrix[user_id]
        user_reviews = user_reviews.toarray().ravel()
        user_rated_items, = np.where(user_reviews > 0)
        user_ratings = user_reviews[user_rated_items]
        return user_ratings


    def _get_item_rating(self, item_id):
        item_reviews = self.sparse_matrix[:, item_id]
        item_reviews = item_reviews.toarray().ravel()
        item_rated_users = np.where(item_reviews > 0)
        item_ratings = item_reviews[item_rated_users]
        return item_ratings


    def _calculate_pseudo_average_user(self, user_id, k=k):
        user_ratings = self._get_user_rating(user_id)
        user_adj = (self.mu * k + np.sum(user_ratings))
        norm_factor = (k + np.size(user_ratings))
        return user_adj / norm_factor


    def _calculate_pseudo_average_item(self, item_id, k=k):
        item_ratings = self._get_item_rating(item_id)
        item_adj = (self.mu * k + np.sum(item_ratings))
        norm_factor = (k + np.size(item_ratings))
        return item_adj / norm_factor


    def _user_bias(self, user_id):
        return self.user_pseudo_rating[user_id] - self.mu


    def _item_bias(self, item_id):
        return self.item_pseudo_rating[item_id] - self.mu


    def _compute_pseudo_averages(self):
        for user in self.user_pseudo_rating.keys():
            self.user_pseudo_rating[user] = self._calculate_pseudo_average_user(user)
        for item in self.item_pseudo_rating.keys():
            self.item_pseudo_rating[item] = self._calculate_pseudo_average_item(item)
        return None
            

    def get_rating(self, user_id, item_id):
        return self.sparse_matrix[user_id, item_id]


    def predict(self, user_id, item_id):
        try:
            return np.dot(self.item_matrix[:, item_id], self.user_matrix[:, user_id])
        except:
            print('user: {userid}, item: {itemid}'.format(userid=user_id, itemid=item_id))
            return np.dot(self.item_matrix[:, item_id], self.user_matrix[:, user_id])
            
            
    def calc_error(self, user_id, item_id):
        r_ij = self.get_rating(user_id, item_id)
        mu = self.mu
        user_bias = self._user_bias(user_id)
        item_bias = self._item_bias(item_id)
        r_hat = self.predict(user_id, item_id)
        return (r_ij - mu - user_bias - item_bias - r_hat)


    def train(self, user_id, item_id, feature_id, alpha=0.005, K=K):
        '''
        '''
        err = self.calc_error(user_id, item_id)

        user_feature_vector = self.user_matrix[feature_id]
        item_feature_vector = self.item_matrix[feature_id]

        user_feature_value = user_feature_vector[user_id]
        item_feature_value = item_feature_vector[item_id]

        user_feature_vector[user_id] += alpha * (err * item_feature_value - K * user_feature_value)
        item_feature_vector[item_id] += alpha * (err * user_feature_value - K * item_feature_value)

        return err ** 2


    def calculate_features(self):
        mse = 0
        last_mse = 0
        n_ratings = np.count_nonzero(self.sparse_matrix.toarray().ravel())
        users, items = self.sparse_matrix.nonzero()
        for feature in xrange(self.rank):
            j = 0
            while (j < self.min_iter) or (mse < last_mse - self.min_imp):
                squared_error = 0
                last_mse = mse

                for user_id, item_id in zip(users, items):
                    squared_error += self.train(user_id, item_id, feature)

                mse = (squared_error / n_ratings)
                print('MSE = {mse}'.format(mse=str(mse)))
                j += 1
                sys.stdout.flush()
            print('Feature = {feature_id}'.format(str(feature)))
        print('Converged')


    @staticmethod
    def _load_data(file_path):
        '''
        Accepts a tab-delimited file with first three columns as
            user, item, value.
        Returns sparse matrix.
        '''
        with open(file_path) as f_in:
            data = np.array([[int(tok) for tok in line.split('\t')[:3]]
                        for line in f_in])
        ij = data[:, :2]
        ij -= 1
        values = data[:, 2]
        sparse_matrix = sparse.csc_matrix((values, ij.T)).astype(float)
        return sparse_matrix


    if __name__ == '__main__':
        # TODO:
        #    * use movielens 100k database to run a sample test
        pass
