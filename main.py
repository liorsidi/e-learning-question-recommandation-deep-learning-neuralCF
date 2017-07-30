from __future__ import division
# load data set create - user question matrix
import random

import numpy as np
import pandas as pd


answers_path= 'algebra_2005_2006' + '\\algebra_2005_2006_train.txt'


questionaire_col = 'unit'
questions_col = 'Step Name'
student_col = 'Anon Student Id'
difficulty_col = 'Difficulty'
#rank_dif_col = 'Rank_Difficulty'


def Ak(L, qk, s1, s2):
    sum_IA = 0
    s1_qk_rank_dif = s1[s1[questions_col] == qk][difficulty_col].iloc[0]
    s2_qk_rank_dif = s2[s2[questions_col] == qk][difficulty_col].iloc[0]

    Zk = s2[(s2[questions_col].isin(L)) & (s2[difficulty_col] <= s2_qk_rank_dif)][questions_col]
    if len(Zk) == 0:
        return 0
    k = len(Zk)

    for qj_name in Zk:
        if s1[s1[questions_col] == qj_name][difficulty_col].iloc[0] <= s1_qk_rank_dif:
            sum_IA += 1
    A_k = sum_IA/k
    return A_k

def SAP(L,s1, s2):
    """

    :param L: list of questions
    :param s1: questions rank original
    :param s2: questions rank proposed
    :return:
    """
    sum_Ak = 0
    s1 = s1[s1[questions_col].isin(L)]
    s2 = s2[s2[questions_col].isin(L)]
    for qk in L:
        sum_Ak += Ak(L,qk,s1,s2)
    return sum_Ak/len(L)

def NDPM(L,s1, s2):
    """

    :param L: list of questions
    :param s1: questions rank original
    :param s2: questions rank proposed
    :return:
    """

    sum_Ak = 0
    s1 = s1[s1[questions_col].isin(L)]
    s2 = s2[s2[questions_col].isin(L)]
    C_plus = 0
    C_minus = 0
    C_u = 0
    C_s = 0
    C_u0 = 0

    for q_ui in range(len(L)):
        for q_uj in range(q_ui,len(L)):
            s1_q_ui_rank_dif = s1[s1[questions_col] == L[q_ui]][difficulty_col].iloc[0]
            s2_q_ui_rank_dif = s2[s2[questions_col] == L[q_ui]][difficulty_col].iloc[0]

            s1_q_uj_rank_dif = s1[s1[questions_col] == L[q_uj]][difficulty_col].iloc[0]
            s2_q_uj_rank_dif = s2[s2[questions_col] == L[q_uj]][difficulty_col].iloc[0]

            C_plus += np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif) * np.sign(s2_q_ui_rank_dif -s2_q_uj_rank_dif)
            C_minus += np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif) * np.sign(s2_q_uj_rank_dif - s2_q_ui_rank_dif)
            C_u += pow(np.sign(s1_q_ui_rank_dif - s1_q_uj_rank_dif),2)
            C_s += pow(np.sign(s2_q_ui_rank_dif -s2_q_uj_rank_dif),2)

    C_u0 += C_u -(C_plus+C_minus)
    NDPM = (C_minus + 0.5 * C_u0) / C_u
    return NDPM

def SAP_update(L,s1, s2, prev_score, new_q,w_q):
    s1 = s1[s1[questions_col].isin(L)]
    s2 = s2[s2[questions_col].isin(L)]
    new_q_ak = Ak(L, new_q, s1, s2)
    w_new_q_ak = new_q_ak * w_q / len(L)
    w_sum_Ak = prev_score * (len(L)+ w_q)/len(L)
    return w_new_q_ak + w_sum_Ak

def relative_voting(qk,ql,S,similarities_i):
    score_k = 0
    students = S[student_col].unique()
    for student_j  in students:
        s_j_q = S[S[student_col] == student_j][questions_col].unique()
        if qk not in s_j_q :
            print "student "  + student_j + " didnt answer question " + qk
            continue
        if ql not in s_j_q :
            print "student "  + student_j + " didnt answer question " + ql
            continue

        s_j = S[S[student_col] == student_j]
        s_j_qk_rank = s_j[s_j[questions_col] == qk][difficulty_col].iloc[0]
        s_j_ql_rank = s_j[s_j[questions_col] == ql][difficulty_col].iloc[0]

        gamma = np.sign(s_j_qk_rank - s_j_ql_rank)
        score_k += similarities_i[student_j] * gamma

    return np.sign(score_k)

def copeland(q, S,similarities_i):
    sum_c = 0
    Li_no_q = S[S[questions_col] != q][questions_col].unique()
    students = similarities_i.keys()
    relevant_students_q = S[(S[student_col].isin(students)) &
                          (S[questions_col] == q)][student_col].unique()
    for ql in Li_no_q:
        relevant_students_ql = S[(S[student_col].isin(relevant_students_q)) &
                          (S[questions_col] == ql)][student_col].unique()
        relevant_S = S[S[student_col].isin(relevant_students_ql)]
        #start_time = time.time()
        sum_c += relative_voting(q,ql,relevant_S,similarities_i)
        #print 'rv time for questions ' + q + ' and ' + ql +': ' + str(round((time.time() - start_time) / 60.0, 3))
    return sum_c

def calc_similarities(Q, S, sim_func):
    similarities = {}
    S_no_q = S[~S[questions_col].isin(Q)]
    for student_i in S[student_col].unique():
        Q_i = S[(S[student_col] == student_i) & (S[questions_col].isin(Q))][questions_col].unique()
        if len(Q_i) == 0:
            continue
        start_time_i = time.time()
        similarities[student_i] = {}
        s_i = S_no_q[S_no_q[student_col] == student_i]
        # iterate student j that have questions for questionaire and from previous questions
        for student_j in S[(S[student_col] != student_i) & (S[questions_col].isin(Q_i))][student_col].unique():
            if student_j in similarities[student_i]:
                similarities[student_i][student_j] =  similarities[student_j][student_i]
                continue
            #start_time_j = time.time()
            s_j = S_no_q[S_no_q[student_col] == student_j]
            #check other questions join
            L = pd.merge(s_j, s_i, how='inner', on=[questions_col])[questions_col].unique()
            if len(L) >1 :
                similarities[student_i][student_j] = sim_func(L,s_i, s_j)
            #print 'similar time for student j ' + student_j + ' : ' + str(round((time.time() - start_time_j) / 60.0, 3))
        #print 'similar time for student i ' + student_i + ' : ' + str(round((time.time() - start_time_i) / 60.0, 3))
    return similarities

def create_similarities(name,Q,S, sim_func, force = False):
    import os
    file_name = 'similarities\\' + questionaire + ' LOO students scores train'
    if not os.path.isfile(file_name) or force:
        similarities = calc_similarities(Q, S, sim_func)
        with open(file_name, 'wb') as f:
            pickle.dump(similarities, f)
    else:
        with open(file_name) as f:
            similarities = pickle.load(f)
    return similarities

def update_similarity(Q,S,similarities_i, new_q, s_i,sim_update_func = SAP_update , w_q=2.0):
    #Q = Q + [new_q]
    students_j = S[(~S[questions_col].isin(Q))&(S[student_col].isin(similarities_i))&(S[questions_col].isin([new_q]))][student_col].unique()
    sim_changes = 0.0
    if len(students_j) == 0:
        #print "no students to update"
        return similarities_i, sim_changes
    #iterate student that ansered new_q
    for student_j in students_j:
        #print "update student " + student_j
        #new student doesnt count
        # if student_j not in similarities_i:
        #     continue
        # start_time_j = time.time()
        s_j = S[S[student_col] == student_j]
        # join all i and j quesions
        L = pd.merge(s_j, s_i, how='inner', on=[questions_col])[questions_col].unique()
        if new_q not in L:
            continue
        prev_score = similarities_i[student_j]
        #updating only 1 question
        similarities_i[student_j] = sim_update_func(L,s_i, s_j, prev_score, new_q, w_q)
        sim_changes += abs(prev_score - similarities_i[student_j])
    avg_changes = sim_changes/len(student_j)
    return similarities_i, sim_changes

def rank_questions_for_student_online(Q,S,s_i,similarities_i,next_question_w, sim_func = SAP,sim_update_func = SAP_update):
    similarities_i = dict(similarities_i)
    Q_remain = list(Q)
    Q_ranked = []
    rank_results = []
    start_time = time.time()

    offline_score = 0.0
    sim_avg_changes = 0.0
    for i in range(len(Q)):
        start_time_i = time.time()

        Q_ranked_i = rank_questions_for_student(Q_remain,S,s_i,similarities_i)
        q_min = Q_ranked_i[Q_ranked_i[difficulty_col] == min(Q_ranked_i[difficulty_col])]

        rank_result_i = {}
        rank_result_i['iteration'] = i
        rank_result_i['questions_rank'] = Q_ranked_i[[questions_col,difficulty_col]].to_dict()
        rank_result_i['similarity_avg_changes'] = sim_avg_changes
        rank_result_i['SAP'] = SAP(Q_remain, S, Q_ranked_i)
        rank_result_i['NDPM'] = NDPM(Q_remain, S, Q_ranked_i)
        rank_result_i['time'] = round((time.time() - start_time_i) / 60.0, 3)
        rank_results.append(rank_result_i)

        if i == 0: offline_score = rank_result_i[sim_func.__name__]

        q_ranked = {}
        q_ranked[student_col] = s_i[student_col].iloc[0]
        q_ranked[questions_col] = q_min[questions_col].iloc[0]
        q_ranked[difficulty_col] = i
        Q_ranked.append(q_ranked)
        Q_remain.remove(q_ranked[questions_col])
        similarities_i, sim_avg_changes = update_similarity(Q_remain, S, similarities_i, q_min[questions_col].iloc[0], s_i, SAP_update, next_question_w)

    Q_ranked = pd.DataFrame(Q_ranked)
    rank_result_full = {}
    rank_result_full['iteration'] = 'full'
    rank_result_full['questions_rank'] = Q_ranked[[questions_col,difficulty_col]].to_dict()
    rank_result_full[sim_func.__name__] = sim_func(Q, S, Q_ranked)
    rank_result_full['time'] = round((time.time() - start_time) / 60.0, 3)
    rank_results.append(rank_result_full)

    print "online improves in " + str(rank_result_full[sim_func.__name__] - offline_score)
    return pd.DataFrame(rank_results)

def rank_questions_for_student(Q, S, s_i, similarities):
    Q_ranked = []

    for q in Q:
        relevant_questions = S[(S[student_col].isin(similarities.keys())) &
                               (S[questions_col].isin(Q))][questions_col].unique()

        relevant_students = similarities.keys() + [s_i[student_col].iloc[0]]
        S_filtered = S[(S[student_col].isin(relevant_students)) &
                       (S[questions_col].isin(relevant_questions))]

        q_ranked = {}
        q_ranked[student_col] = s_i[student_col].iloc[0]
        q_ranked[questions_col] = q

        q_ranked[difficulty_col] = copeland(q, S_filtered, similarities)

        Q_ranked.append(q_ranked)
    return pd.DataFrame(Q_ranked)

def gold_standard_score(answer_stats):
    if answer_stats['Correct First Attempt'] == 1: return 1
    return 1 - 0.2* answer_stats['Incorrects']

def get_ranked_questions(path):
    df = pd.read_table(path)
    df['Difficulty'] = df.apply(lambda row: gold_standard_score(row), axis =1)
    return df[[student_col,'Problem Hierarchy','Problem Name','Step Name','Difficulty']]


import pickle
import time
import os

#S - all ranked questions of all stidents
#Q - all questions in questionaire
#similarities - the similarities between students without Q
#Q_i - questions answerd bu student i
#s_i - ranked quesrtions of student i

#students - all stu
S = get_ranked_questions(answers_path)
problem_df = S['Problem Hierarchy'].apply(lambda x: pd.Series([i for i in reversed(x.split(','))]))
problem_df.rename(columns={1:'unit',0:'quest'},inplace=True)
S = pd.concat((problem_df,S),axis=1)
questionaires = S[questionaire_col].unique()
online_results = pd.DataFrame()

next_question_ws = [1,2,4]


for questionaire in questionaires:
    print "start evaluate questionaire: " + questionaire

    # res_file_name = 'results\\' + 'LOO students scores' + questionaire + ".csv"
    # if os.path.isfile(res_file_name):
    #      continue
    Q =  list(S.loc[S[questionaire_col] == questionaire][questions_col].unique())
    students_Q = S[S[questionaire_col] == questionaire][student_col].unique()
    #start_time = time.time()
    similarities = create_similarities(questionaire,Q,S[S[student_col].isin(students_Q)], SAP)
    similarities = {key: value for key, value in similarities.items() if value != {}}

    # keep only students with similarity and keep random from them
    students_Q = similarities.keys()
    if len(students_Q) < 3:
        print 'not enough similarities'
        continue
    random_students = [random.randint(0, len(students_Q)-1) for r in xrange(10)]
    random_students = [students_Q[s] for s in random_students]
    #random_students = students_Q # all students!!
    scores = []
    for student_i in random_students:
        for next_question_w in next_question_ws:
            print "start student " + student_i
            Q_i = list(S[(S[student_col] == student_i) & (S[questions_col].isin(Q))][questions_col].unique())
            #if len(Q_i) == 0:
            #    continue
            s_i = S[S[student_col] == student_i]
            #ranked_questions = rank_questions_for_student(Q_i, S, s_i, similarities[student_i])
            #sap_i = SAP(Q_i, S, ranked_questions_online)
            #scores.append(sap_i)
            online_results_i_q = rank_questions_for_student_online(Q_i, S, s_i,similarities[student_i],next_question_w)
            online_results_i_q[student_col] = student_i
            online_results_i_q[questionaire_col] = questionaire
            online_results_i_q['next_q_w'] = next_question_w

            online_results = pd.concat([online_results,online_results_i_q])

    students_selected = len(random_students)
    del similarities

    online_results.to_csv('results\\' + str(next_question_w) + "_" + str(students_selected) + 'LOO students scores train.csv')



