from __future__ import division

import Constants as C
import random

import numpy as np
import pandas as pd
import time
import operator
from RankEvaluation import SAP, SAP_update

def relative_voting(qk, ql, S, similarities_i):
    score_k = 0
    students = S[C.student_col].unique()
    for student_j in students:
        s_j = S[S[C.student_col] == student_j]
        score_k += similarities_i[student_j] * np.sign(s_j[s_j[C.questions_col] == ql][C.difficulty_col].iloc[0] -
                                                       s_j[s_j[C.questions_col] == qk][C.difficulty_col].iloc[0])

        if (abs(score_k) > 3):
            break

    return np.sign(score_k)

class Edurank(object):
    def __init__(self, s_id,Q_i, Q,memory_size=5, sim_func=SAP,sim_update_func=SAP_update):
        """

        :param s_id: student id
        :param S: all students answers
        :param questionaire: questionaires
        """
        self.s_id = s_id
        self.Q = Q
        self.Q_i = Q_i
        self.sim_func = sim_func
        self.sim_update_func = sim_update_func
        self.similarities = {}
        self.memory_size = memory_size
        self.similarities_top = {}
        self.S_top = pd.DataFrame()

        self.params = dict(memory_size=self.memory_size,
                           sim_func=self.sim_func,
                           )

    def rate(self):
        Q_ranked = []
        ranks = []

        similarities = self.similarities_top
        S = self.S_top
        rvs = {}

        #compute a relative voting in memory
        for q1 in xrange(len(self.Q_i)):

            students = similarities.keys()

            relevant_students_q = S[(S[C.student_col].isin(students)) &
                                    (S[C.questions_col] == self.Q_i[q1])][C.student_col].unique()

            Li_no_q = S[(S[C.questions_col] != self.Q_i[q1]) & (S[C.student_col].isin(relevant_students_q))][
                C.questions_col].unique()

            rvs[q1] = {}
            print q1

            for q2 in xrange(q1,len(self.Q_i)):
                if self.Q_i[q2] not in Li_no_q:
                    continue
                relevant_students_ql = S[(S[C.student_col].isin(relevant_students_q)) &
                                         (S[C.questions_col] == self.Q_i[q2])][C.student_col].unique()

                if len(relevant_students_ql) == 0:
                    continue
                relevant_S = S[S[C.student_col].isin(relevant_students_ql)]
                rvs[q1][q2] = relative_voting(self.Q_i[q1], self.Q_i[q2], relevant_S, similarities)

                if q2 not in rvs:
                    rvs[q2] = {}
                rvs[q2][q1] = rvs[q1][q2]

        # compute a copeland score based on the rv
        for q1 in xrange(len(self.Q_i)):
            sum_copeland = 0
            if q1 in rvs:
                for q2 in xrange(q1, len(self.Q_i)):
                    if q2 in rvs[q1]:
                        sum_copeland += rvs[q1][q2]

            q_ranked = {}
            q_ranked[C.student_col] = self.s_id
            q_ranked[C.questions_col] = self.Q_i[q1]
            q_ranked[C.difficulty_col] = sum_copeland
            Q_ranked.append(q_ranked)

        return pd.DataFrame(Q_ranked)

    def update_model(self, new_Q):
        self.S_top = self.S_top.append(new_Q,ignore_index=True)
        s_i = self.S_top[self.S_top[C.student_col] == self.s_id]
        self.Q = [x for x in self.Q if x not in new_Q[C.questions_col].unique()]

        # iterate student that ansered new_q
        for student_j in self.similarities_top.keys():
            s_j = self.S_top[self.S_top[C.student_col] == student_j]
            # join all i and j quesions
            L = pd.merge(s_j, s_i, how='inner', on=[C.questions_col])[C.questions_col].unique()
            for new_q in new_Q[C.questions_col].unique():
                if new_q not in L:
                    continue
                prev_score = self.similarities_top[student_j]
                # updating only 1 question
                self.similarities_top[student_j] = self.sim_update_func(L, s_i, s_j, prev_score, new_q)
                print "sim changes:"
                print abs(prev_score - self.similarities_top[student_j])

    def fit(self,S):
        self.similarities = {}

        s_i = S[S[C.student_col] == self.s_id]
        # iterate student j that have questions for questionaire and from previous questions
        for student_j in S[(S[C.student_col] != self.s_id) & (S[C.questions_col].isin(self.Q))][C.student_col].unique():
            self.similarities[student_j] = 0
            counter = 0
            s_j = S[S[C.student_col] == student_j]
            # check other questions join
            join_questionaires = pd.merge(s_j, s_i, how='inner', on=[C.questionaire_col])[C.questionaire_col].unique()
            for qusare in join_questionaires:
                s_j_q = s_j[s_j[C.questionaire_col] == qusare]
                s_i_q = s_i[s_i[C.questionaire_col] == qusare]
                L = pd.merge(s_j_q, s_i_q, how='inner', on=[C.questions_col])[C.questions_col].unique()
                if len(L) > 1:
                    self.similarities[student_j] += self.sim_func(L, s_i_q, s_j_q)
                    counter += 1
            if counter == 0:
                del self.similarities[student_j]
            else:
                self.similarities[student_j] = self.similarities[student_j] / counter

        if not self.similarities:
            raise Exception('studnt dont share items with others')
        self.similarities = {key: value for key, value in self.similarities.items() if value != {}}

        self.S_top = S[S[C.questions_col].isin(self.Q_i)]
        similiarities_relevant = {your_key: self.similarities[your_key] for your_key in self.S_top[self.S_top[C.student_col].isin(self.similarities.keys())][C.student_col].unique()}
        self.similarities_top = dict(
            sorted(similiarities_relevant.iteritems(), key=similiarities_relevant.get, reverse=True)[:self.memory_size ])


