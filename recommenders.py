import pandas as pd
import numpy as np
import scipy.spatial.distance as scdist
import operator
# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='sales_value', # Можно пробоват ьдругие варианты
                                          aggfunc='count', 
                                          fill_value=0
                                         )
        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(user_item_matrix_,user, model, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        user_item_matrix_ = MainRecommender.prepare_matrix(user_item_matrix_)
        user_top = user_item_matrix_[user_item_matrix_.index==user].T[user_item_matrix_[user_item_matrix_.index==user].T[user]>0].index[:N]
    #     print(user_top)
        top_similar_items=[]
        for i in user_top:
            sim_items = user_item_matrix_.T.iloc[user_item_matrix_.T.index==i,:]
            sim = 1-scdist.cdist(user_item_matrix_.T, sim_items, 'yule')
            top_similar_items=top_similar_items+sorted([[k,i[0]] for k,i in zip(user_item_matrix_.T.index,sim)], key=operator.itemgetter(1, 0))
        top_similar_items = [int(i) for i in list(np.array(sorted([[k,i[0]] for k,i in zip(user_item_matrix_.T.index,sim)], key=operator.itemgetter(1, 0))[-N:])[:,0])]
    #     print(top_similar_items)

        assert len(top_similar_items[:N]) == N, 'Количество рекомендаций != {}'.format(N)
        return top_similar_items
    
    
    def get_similar_users_recommendation(user_item_matrix,user, model, N=5,n_users = 20):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        user_item_matrix = MainRecommender.prepare_matrix(user_item_matrix)
        id_to_itemid, id_to_userid, itemid_to_id, userid_to_id = MainRecommender.prepare_dicts(user_item_matrix)
        user_vector = user_item_matrix.iloc[user_item_matrix.index==user,:]
        sim = 1-scdist.cdist(user_item_matrix, user_vector, 'cosine')
        top_similar_users = [k for k,i in sorted([[k,i] for k,i in zip(user_item_matrix.index,sim)], key=operator.itemgetter(1, 0))]
        res = [id_to_itemid[rec[0]] for rec in 
                model.recommend(userid=userid_to_id[user],user_items=csr_matrix(user_item_matrix.T[top_similar_users].T).tocsr(),N=N,filter_already_liked_items=False,filter_items=[itemid_to_id[999999]],recalculate_user=True)]


        assert len(res[:N]) == N, 'Количество рекомендаций != {}'.format(N)
        return res

