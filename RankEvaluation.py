from __future__ import division

import threading

import logging

import Constants as C
import numpy as np
import pandas as pd

def Ak(L, qk, s1, s2):
    sum_IA = 0
    s1_qk_rank_dif = s1[s1[C.questions_col] == qk][C.difficulty_col].iloc[0]
    s2_qk_rank_dif = s2[s2[C.questions_col] == qk][C.difficulty_col].iloc[0]

    Zk = s2[(s2[C.questions_col].isin(L)) & (s2[C.difficulty_col] <= s2_qk_rank_dif)][C.questions_col].unique()
    if len(Zk) == 0:
        return 0
    k = len(Zk)

    A_k = len(s1[(s1[C.questions_col].isin(Zk)) & (s1[C.difficulty_col] <= s1_qk_rank_dif)])#[C.questions_col].unique())
    k = len(s1[(s1[C.questions_col].isin(Zk))])
    # for qj_name in Zk:
    #     if s1[s1[C.questions_col] == qj_name][C.difficulty_col].iloc[0] <= s1_qk_rank_dif:
    #         sum_IA += 1
    #
    #
    # A_k = sum_IA/k
    return A_k / k

def SAP(L,s1, s2):
    """

    :param L: list of questions
    :param s1: questions rank original
    :param s2: questions rank proposed
    :return:
    """
    sum_Ak = 0
    s1 = s1[s1[C.questions_col].isin(L)]
    s2 = s2[s2[C.questions_col].isin(L)]
    for qk in L:
        sum_Ak += Ak(L,qk,s1,s2)
    return sum_Ak/len(L)

def SAP_update(L,s1, s2, prev_score, new_q,w_q=0.2):
    s1 = s1[s1[C.questions_col].isin(L)]
    s2 = s2[s2[C.questions_col].isin(L)]
    new_q_ak = Ak(L, new_q, s1, s2)
    # w_new_q_ak = new_q_ak * w_q / len(L)
    # w_sum_Ak = prev_score * (len(L)+ w_q)/len(L)
    #return w_new_q_ak + w_sum_Ak
    final_score = ((1-w_q) * prev_score) + (w_q * new_q_ak)
    return final_score

def NDPM(L,s1, s2):
    """

    :param L: list of questions
    :param s1: questions rank original
    :param s2: questions rank proposed
    :return:
    """

    sum_Ak = 0
    s1 = s1[s1[C.questions_col].isin(L)]
    s2 = s2[s2[C.questions_col].isin(L)]
    C_plus = 0
    C_minus = 0
    C_u = 0
    C_s = 0
    C_u0 = 0

    for q_ui in range(len(L)):
        for q_uj in range(q_ui,len(L)):
            s1_q_ui_rank_dif = s1[s1[C.questions_col] == L[q_ui]][C.difficulty_col].iloc[0]
            s2_q_ui_rank_dif = s2[s2[C.questions_col] == L[q_ui]][C.difficulty_col].iloc[0]

            s1_q_uj_rank_dif = s1[s1[C.questions_col] == L[q_uj]][C.difficulty_col].iloc[0]
            s2_q_uj_rank_dif = s2[s2[C.questions_col] == L[q_uj]][C.difficulty_col].iloc[0]

            C_plus += np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif) * np.sign(s2_q_ui_rank_dif -s2_q_uj_rank_dif)
            C_minus += np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif) * np.sign(s2_q_uj_rank_dif - s2_q_ui_rank_dif)
            C_u += pow(np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif),2)
            C_s += pow(np.sign(s2_q_ui_rank_dif -s2_q_uj_rank_dif),2)

    C_u0 += C_u -(C_plus+C_minus)
    NDPM = (C_minus + 0.5 * C_u0) / C_u
    return NDPM

class summer(object):
    def __init__(self, start=0):
        self.lock = threading.Lock()
        self.value = start

    def increment(self,add_val):
        logging.debug('Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.value = self.value + add_val
        finally:
            self.lock.release()

def calc_pair(s1,s2,q_ui,q_uj,C_plus,C_minus,C_u,C_s):
    s1_q_ui_rank_dif = s1[s1[C.questions_col] == q_ui][C.difficulty_col].iloc[0]
    s2_q_ui_rank_dif = s2[s2[C.questions_col] == q_ui][C.difficulty_col].iloc[0]

    s1_q_uj_rank_dif = s1[s1[C.questions_col] == q_uj][C.difficulty_col].iloc[0]
    s2_q_uj_rank_dif = s2[s2[C.questions_col] == q_uj][C.difficulty_col].iloc[0]

    C_plus.increment(np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif) * np.sign(s2_q_ui_rank_dif - s2_q_uj_rank_dif))
    C_minus.increment(np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif) * np.sign(s2_q_uj_rank_dif - s2_q_ui_rank_dif))
    C_u.increment(pow(np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif), 2))
    C_s.increment(pow(np.sign(s2_q_ui_rank_dif - s2_q_uj_rank_dif), 2))

def NDPM2(L,s1, s2):
    """

    :param L: list of questions
    :param s1: questions rank original
    :param s2: questions rank proposed
    :return:
    """

    sum_Ak = 0
    s1 = s1[s1[C.questions_col].isin(L)]
    s2 = s2[s2[C.questions_col].isin(L)]
    C_plus = summer()
    C_minus = summer()
    C_u = summer()
    C_s = summer()
    C_u0 = 0
    total_t = 8
    t_amount = 0
    for q_ui in range(len(L)):
        for q_uj in range(q_ui,len(L)):
            t = threading.Thread(target=calc_pair, args=(s1,s2,L[q_ui],L[q_uj] ,C_plus,C_minus,C_u,C_s,))
            t.start()
            t_amount+=1
            if t_amount>=total_t:
                logging.debug('Waiting for worker threads')
                main_thread = threading.currentThread()
                for t in threading.enumerate():
                    if t is not main_thread:
                        t.join()
                t_amount = 0

    C_u0 += C_u.value -(C_plus.value +C_minus.value)
    NDPM = (C_minus.value + 0.5 * C_u0) / C_u.value
    return NDPM

def gold_standard_score(answer_stats):
    if answer_stats['Correct First Attempt'] == 1: return 1
    return 1 - 0.2* answer_stats['Incorrects']

def get_ranked_questions(df):
    df[C.difficulty_col] = df.apply(lambda row: gold_standard_score(row), axis =1)
    return df
