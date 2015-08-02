from __future__ import division

import numpy as np
from scipy import sparse
import sys


class PMF:
    '''
    Notation:
    ------------
        * f number of latent features
        * m users
        * n items
        * r[u,i] indicates the preference for user u of item i
        * r_hat[u,i]  indicates the predicted preference for user
            u of item i
        * K indicates {(u,i) | r[u,i] is known}
        * b_u is user bias
        * b_i is item bias
        * q is a low-rank item matrix with f latent features
        * p is a low-rank user matrix with f latent features
        * min_iter is the min number of iterations for descent
        * max_iter is the max number of iterations for descent
        * min_improvement is the minimum improvement needed to continue
            training the current feature

    Goal:
    ------------
    We're going to use previously observed ratings to fit 
    a rating estimator model.
    '''

    def __init__(self, file_path, f, gamma=0.005, lambda_=0.02, min_iter = 1, max_iter=10, 
                 min_improvement=1e-4, **kwargs):
        self.R_sparse = self._load_data(file_path)
        self.R = self.R_sparse.todense()
        self.f = f
        self.gamma = gamma
        self.lambda_ = lambda_
        self.K_users, self.K_items = self.R_sparse.nonzero()

        n = self.R.shape[0]
        m = self.R.shape[1]

        if hasattr(self, 'p'):
            self.p = p
        else:
            self.p = np.ones((f, n)) * np.random.uniform(-0.01, 0.01, size=f*n).reshape(f,n)
        if hasattr(self, 'q'):
            self.q = q
        else:
            self.q = np.ones((f, m)) * np.random.uniform(-0.01, 0.01, size=m*f).reshape(f,m)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.min_improvement = min_improvement

        self.get_baseline()


    def get_rating(self, u, i):
        return self.R[u,i]


    def predict_rhat(self, u, i):
        return self.mu + self.b_i[i] + self.b_u[u] + np.dot(self.q.T[i,:], self.p[:,u])


    def _get_mu(self, x):
        return np.nansum(x) / np.count_nonzero(~np.isnan(x))


    def get_baseline(self):
        self.mu = self._get_mu(self.R)
        self.b_u = self.mu - [self._get_mu(self.R[u,:]) for u in xrange((self.R.shape[0]))]
        self.b_i = self.mu - [self._get_mu(self.R[:,i]) for i in xrange((self.R.shape[1]))]
        return None

    def baseline(self, matrix):
        '''
        Returns the global average rating, user bias, and item bias.
        '''
        mu = self._get_mu(matrix)
        user_bias = mu - [self._get_mu(matrix[i,:]) for i in xrange((matrix.shape[0]))]
        item_bias = mu - [self._get_mu(matrix[:,i]) for i in xrange((matrix.shape[1]))]
        return None


    def update(self, u, i, err):
        self.b_u[u] = self.b_u[u] + self.gamma * (err - self.lambda_ * self.b_u[u])
        self.b_i[i] = self.b_i[i] + self.gamma * (err - self.lambda_ * self.b_i[i])
        self.q[:,i] = self.q[:,i] + self.gamma  * (err * self.p[:,u] - self.lambda_ * self.q[:,i])
        self.p[:,u] = self.p[:,u] + self.gamma * (err * self.q[:,i] - self.lambda_ * self.p[:,u])
        return None
    

    def get_error(self, u, i):
        r = self.get_rating(u, i)
        r_hat = self.predict_rhat(u, i)
        err = (r - r_hat)**2 + self.lambda_ * \
              (self.b_i[i]**2 + self.b_u[u]**2 + np.linalg.norm(self.q[:,i])**2 + \
                                           np.linalg.norm(self.p[:,u])**2)
        return err


    def train(self):
        rmse = 2.0
        n_ratings = len(self.K_items)

        for feature in xrange(self.f):
            print('--- Calculating Feature: {feat} ---'.format(feat=feature))


            for n in xrange(self.max_iter):
                sq_err = 0.0
                rmse_last = rmse

                for u, i in zip(self.K_users, self.K_items): 
                    try:
                        err = self.get_error(u, i)
                        sq_err = err**2
                        self.update(u, i, err)
                    except:
                        print('u:', u, 'i:', i)
                        print err
                        print rmse_last
                        err = self.get_error(u, i)
                        sq_err = err**2
                        self.update(u, i, err)
                        
                rmse = N.sqrt(sq_err / n_ratings)

                if (n >= self.min_iter and rmse > rmse_last - self.min_improvement):
                    break


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
    test = PMF('./data/ml-100k/u1.base')
    tmp.train()
