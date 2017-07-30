# load data set create - user question matrix
from mpmath import sign
import pandas as pd

answers_path = 'algebra_2005_2006' + '\\algebra_2005_2006_master.txt'

questionaire = 'Unit CTA1_27'
questionaire_col = 'unit'
questions_col = 'Step Name'
student_col = 'Anon Student Id'
rank_dif = 'Rank_Difficulty'


def SAP(L, s_i, s_j):
    sum_Ia = 0
    for qj_index in range(0, len(L)):
        for qk_index in range(qj_index + 1, len(L)):  # TODO check +1
            qj = L[qj_index]
            qk = L[qk_index]
            if qj == qk:
                continue
            s_i_qk_rank = s_i[s_i[student_col] == qk][rank_dif]
            s_i_qj_rank = s_i[s_i[student_col] == qj][rank_dif]
            s_j_qk_rank = s_j[s_j[student_col] == qk][rank_dif]
            s_j_qj_rank = s_j[s_j[student_col] == qj][rank_dif]

            if ((s_i_qk_rank > s_i_qj_rank and s_j_qk_rank > s_j_qj_rank) or
                    (s_i_qk_rank < s_i_qj_rank and s_j_qk_rank < s_j_qj_rank) or
                    (s_i_qk_rank == s_i_qj_rank and s_j_qk_rank == s_j_qj_rank)):
                sum_Ia += 1
    return (1 / (len(L) - 1)) * sum_Ia


def SAP(L,s1, s2):
    sum_Ak = 0
    for qk_index in range(0,len(L)):
        sum_Ak += Ak(L,qk_index,s1,s2)
        for qj_index in range(qk_index+1,len(L)): #TODO check +1

            qj = L[qj_index]
            qk = L[qk_index]
            if qj == qk:
                continue
            s_i_qk_rank = s_i[s_i[questions_col] == qk][rank_dif].iloc[0]
            s_i_qj_rank = s_i[s_i[questions_col] == qj][rank_dif].iloc[0]
            s_j_qk_rank = s_j[s_j[questions_col] == qk][rank_dif].iloc[0]
            s_j_qj_rank = s_j[s_j[questions_col] == qj][rank_dif].iloc[0]

            if ((s_i_qk_rank > s_i_qj_rank and s_j_qk_rank > s_j_qj_rank) or
                (s_i_qk_rank < s_i_qj_rank and s_j_qk_rank < s_j_qj_rank) or
                (s_i_qk_rank == s_i_qj_rank and s_j_qk_rank == s_j_qj_rank)):
                sum_Ia +=1
    Ak = sum_Ia
    return (1/(len(L) -1))*sum_Ia

def get_ranked_difficulties(s):
    s[rank_dif_col] = s['Difficulty'].rank(method='dense', ascending=False)
    return s

def get_join_ranked_difficulties(s_j, s_i):
    # L = all combine joined of students i and j
    L = pd.merge(s_j, s_i, how='inner', on=[questions_col])[questions_col].unique()

    # s_i_j is the stident i join questions with j
    s_i_j = s_i[s_i[questions_col].isin(L)]  # TODO check if s changes
    s_i_j = get_ranked_difficulties(s_i_j)

    s_j = s_j[s_j[questions_col].isin(L)]
    s_j = get_ranked_difficulties(s_j)

    return s_i_j, s_j, L

def calc_similarities(s_i, S, dist_func):
    distances = {}
    s_i = S[S[student_col] == s_i]
    for student_j in S[S[student_col] != s_i][student_col].unique():
        s_j = S[S[student_col] == student_j]

        # L = all combine joined of students i and j
        L = pd.merge(s_j, s_i, how='inner', on=[questions_col])[questions_col].unique()

        # s_i_j is the stident i join questions with j
        s_i_j = s_i[s_i[questions_col].isin(L)]  # TODO check if s changes
        s_i_j[rank_dif] = s_i_j['Difficulty'].rank(method='dense', ascending=False)

        s_j = s_j[s_j[questions_col].isin(L)]
        s_j[rank_dif] = s_j['Difficulty'].rank(method='dense', ascending=False)

        distances[student_j] = dist_func(L, s_i, s_j)
    return distances


def relative_voting(qk, ql, S, similarities):
    score_k = 0
    for student_j, student_j_distance in similarities.iteritems():
        if ql and qk not in S[S[student_col] == student_j][questions_col]:
            continue
        s_j = S[S[student_col] == student_j]
        s_j_qk_rank = s_j[s_j[student_col] == qk]['rank_Difficulty']
        s_j_ql_rank = s_j[s_j[student_col] == ql]['rank_Difficulty']
        gamma = 0
        if s_j_qk_rank == s_j_ql_rank:
            continue
        elif s_j_qk_rank < s_j_ql_rank:
            gamma = -1
        else:
            gamma = 1

        score_k += student_j_distance * gamma

    return sign(score_k)


def copeland(q, S, similarities):
    sum_c = 0
    Li_no_q = S[S[questions_col] != q][questions_col]
    for ql in Li_no_q:
        sum_c += relative_voting(q, ql, S, similarities)


def rank_questions_for_student(Q, S, s_i):
    similarities = calc_similarities(s_i, S, SAP)
    copland_scores = {}
    for q in Q:
        copland_scores[q] = copeland(q, S, similarities)
    return copland_scores


class Edurank(object):
    def __init__(self, students_questions, student_i, qestions):
        self.students_questions = students_questions
        self.student_i = student_i
        self.qestions = qestions
        self.students_similarity = calc_students_distance(student_i, students_questions, SAP)

    def rank_questions_for_student(self):
        # Li - test questions
        # S - all students rankings
        #

        # Li = self.students_questions_ranking.loc[self.students_questions_ranking[student_col] == self.student_i][
        #     questions_col].unique()
        copland_scores = {}
        for q in self.qestions:
            copland_scores[q] = copeland(q, self.students_questions, self.student_i)
        return copland_scores

    def online_rank_questions(self, questions_to_rank, student_i):
        rv_list = []
        questions_left = questions_to_rank

        for q_iter in questions_to_rank:
            experiment_questions_ranking = copy(self.students_questions_ranking)
            experiment_questions_ranking[student_i][questions_left] = None
            students_similarity = cf_sAP(experiment_questions_ranking)
            rv = relative_voting(students_similarity, experiment_questions_ranking, student_i, questions_left)
            rv_list.append(copeland(relative_voting(relative_voting))[0])
            questions_left = copeland(relative_voting(relative_voting))[1:]
        return rv_list


def gold_standard_score(answer_stats):
    if answer_stats['Correct First Attempt'] == 1: return 1
    return 1 - 0.2 * answer_stats['Incorrects']


def get_ranked_questions(path):
    df = pd.read_table(path)
    df['Difficulty'] = df.apply(lambda row: gold_standard_score(row), axis=1)
    return df[[student_col, 'Problem Hierarchy', 'Problem Name', 'Step Name', 'Difficulty']]


def ap_score(list_a, list_b):
    pass


def ndpm_score(list_a, list_b):
    pass


students_questions_ranking = get_ranked_questions(answers_path)
foo = lambda x: pd.Series([i for i in reversed(x.split(','))])
problem_df = students_questions_ranking['Problem Hierarchy'].apply(foo)
problem_df.rename(columns={1: 'unit', 0: 'quest'}, inplace=True)
# problem_df['unit','quest'] = problem_df[['unit','quest']]
students_questions_ranking = pd.concat((problem_df, students_questions_ranking), axis=1)

questions = students_questions_ranking.loc[students_questions_ranking[questionaire_col] == questionaire][
    questions_col].unique()
students = students_questions_ranking[student_col].unique()
scores = []
for student_i in students:
    ranked_questions = rank_questions_for_student(questions, students_questions_ranking, student_i)

    edurank = Edurank(students_questions_ranking)
    ranked_questions = edurank.rank_questions_for_student(questions, student)
    scores.append(
        (
            ap_score(students_questions_ranking[student], ranked_questions),
            ndpm_score(students_questions_ranking[student], ranked_questions)
        ))

edurank_ranking_i
golden_ranking_i

# calculate difficulty score for each user
# first attempt -> number attamps -> elaps time





# load data set create - user question matrix
import numpy as np
import pandas as pd
answers_path= 'algebra_2005_2006' + '\\algebra_2005_2006_master.txt'

questionaire = 'Unit CTA1_27'
questionaire_col = 'unit'
questions_col = 'Step Name'
student_col = 'Anon Student Id'
difficulty_col = 'Difficulty'
#rank_dif_col = 'Rank_Difficulty'


def Ak(L, qk_index, s1, s2):
    sum_IA = 0
    qk = L[qk_index]
    qk_rank_dif = qk[rank_dif_col].iloc[0]

    Zk = s2[s2[rank_dif_col] > qk_rank_dif][questions_col]
    k = len(Zk)

    for qj_name in Zk:
        if s1[s1[questions_col] == qj_name][rank_dif_col].iloc[0] > qk_rank_dif:
            sum_IA+=1
    return (1/(k-1))*sum_IA

def SAP(L,s1, s2):
    sum_Ak = 0
    for qk_index in range(0,len(L)):
        sum_Ak += Ak(L,qk_index,s1,s2)
    return (1/(len(L) -1))*sum_Ak

def calc_similarities(s_i, S,dist_func):
    distances = {}
    s_i = S[S[student_col] == s_i]
    students_j =S[~S[student_col].isin(s_i[student_col])][student_col].unique()
    for student_j in students_j:
        s_j = S[S[student_col] == student_j]
        #s_i_j, s_j, L = get_join_ranked_difficulties(s_j, s_i)
        L = pd.merge(s_j, s_i, how='inner', on=[questions_col])[questions_col].unique()
        if len(L) >1 :
            distances[student_j] = dist_func(L,s_i, s_j)
    return distances

def relative_voting(qk,ql,S,similarities):
    score_k = 0
    for student_j, student_j_distance  in similarities.iteritems():
        s_j_q = S[S[student_col] == student_j][questions_col].unique()
        if qk not in s_j_q :
            print "student j didnt answer question " + qk
            continue
        if ql not in s_j_q :
            print "student j didnt answer question " + ql
            continue

        s_j = S[S[student_col] == student_j]
        s_j = get_ranked_difficulties(s_j)
        s_j_qk_rank = s_j[s_j[questions_col] == qk][rank_dif_col].iloc[0]
        s_j_ql_rank = s_j[s_j[questions_col] == ql][rank_dif_col].iloc[0]

        gamma = np.sign(s_j_qk_rank - s_j_ql_rank)
        score_k += student_j_distance * gamma

    return np.sign(score_k)

def copeland(q, S,similarities):
    sum_c = 0
    Li_no_q = S[S[questions_col] != q][questions_col]

    for ql in Li_no_q:
        sum_c += relative_voting(q,ql,S,similarities)
    return sum_c

def rank_questions_for_student(Q,S,s_i):
    similarities = calc_similarities(s_i, S, SAP)
    s_i_ranked = {}
    s_i_ranked[student_col] = []
    s_i_ranked[questions_col] = []
    s_i_ranked['Difficulty'] = []
    for q in Q:
        s_i_ranked[student_col] = s_i
        s_i_ranked[questions_col] = q
        s_i_ranked['Difficulty'] = copeland(q,S,similarities)
    return s_i_ranked

def gold_standard_score(answer_stats):
    if answer_stats['Correct First Attempt'] == 1: return 1
    return 1 - 0.2* answer_stats['Incorrects']

def get_ranked_questions(path):
    df = pd.read_table(path)
    df['Difficulty'] = df.apply(lambda row: gold_standard_score(row), axis =1)
    return df[[student_col,'Problem Hierarchy','Problem Name','Step Name','Difficulty']]

def ap_score(list_a, list_b):
    pass

def ndpm_score(list_a, list_b):
    pass

S = get_ranked_questions(answers_path)
foo = lambda x: pd.Series([i for i in reversed(x.split(','))])
problem_df = S['Problem Hierarchy'].apply(foo)
problem_df.rename(columns={1:'unit',0:'quest'},inplace=True)
S = pd.concat((problem_df,S),axis=1)
Q =  S.loc[S[questionaire_col] == questionaire][questions_col].unique()
students =  S[student_col].unique()
scores = []
for s_i in students:
    ranked_questions = rank_questions_for_student(Q, S, s_i)
    ranked_questions_df = pd.DataFrame(ranked_questions)

    scores.append(
        (
            SAP(Q, S, s_i),
            ndpm_score(students_questions_ranking[s_i], ranked_questions)
        ))




edurank_ranking_i
golden_ranking_i

#calculate difficulty score for each user
#first attempt -> number attamps -> elaps time


