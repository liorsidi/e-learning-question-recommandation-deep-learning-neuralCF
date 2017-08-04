from __future__ import division

import os

import itertools

from keras.callbacks import EarlyStopping
from scipy.stats import spearmanr

import Constants as C
import random

import pandas as pd

from Edurank import Edurank
from RankEvaluation import Ak, get_ranked_questions, SAP, NDPM
import time

from NCF import NCF


def get_index_product(params):
    i = 0
    params_index = {}
    for k, v in params.items():
        params_index[k] = i
        i += 1
    params_list = [None] * len(params_index.values())
    for name, loc in params_index.items():
        params_list[loc] = params[name]

    params_product = list(itertools.product(*params_list))
    params_product_dicts = []
    for params_value in params_product:
        params_dict = {}
        for param_name, param_index in params_index.items():
            params_dict[param_name] = params_value[param_index]
        params_product_dicts.append(params_dict)

    return params_product_dicts


def generate_entity(model_class, model_params):
    """
    generate all possible combination of the class with the parmeters
    :param model_class:
    :param model_params:
    :return:
    """
    models = []
    model_params_product = get_index_product(model_params)
    for model_param in model_params_product:
        models.append(model_class(**model_param))
    return models


def get_rate_results(Q_i, standard_rate, rank_result1,rank_result2=None):
    rank_result = {}
    rank_result['total_questions'] = len(Q_i)
    rank_result['SAP_offline'] = SAP(Q_i, standard_rate, rank_result1)
    rank_result['spearman_offline'], p = spearmanr(standard_rate[C.difficulty_col], rank_result1[C.difficulty_col],nan_policy='omit')
    # rank_result['NDPM_offline'] = NDPM(Q_i, S, S_offline)

    if rank_result2 is not None:
        rank_result['SAP_online'] = SAP(Q_i, standard_rate, rank_result2)
        rank_result['SAP_improve'] = rank_result['SAP_online'] - rank_result['SAP_offline']
        rank_result['spearman_online'], p = spearmanr(standard_rate[C.difficulty_col], rank_result2[C.difficulty_col],nan_policy='omit')
        rank_result['spearman_improve'] = rank_result['spearman_online'] - rank_result['spearman_offline']
        # rank_result['NDPM_online'] = NDPM(Q_i, S, S_online)
        # rank_result['NDPM_improve'] = rank_result['NDPM_online'] - rank_result['NDPM_offline']

    return rank_result


def offline_evaluate(sequencer_class, model_params, amount, results_path, amount_q=5, amount_s=3):
    """
    Evaluate the quesiotn sequence model on the Algebra dataset
    :param sequencer_class:
    :param model_params: dictionary with values of lists of parameters
    :param amount: size of dataset to filter
    :param results_path:
    :param amount_q: number of questionairs to evaluate
    :param amount_s: number of students to evaluate
    :return:
    """
    S = get_data(amount)
    questionaires = get_questionaires('fixed',S)[:amount_q]

    all_results = []
    for questionaire in questionaires:
        print "start evaluate questionaire: " + questionaire
        Q = list(S.loc[S[C.questionaire_col] == questionaire][C.questions_col].unique())
        students_i = S[S[C.questionaire_col] == questionaire][C.student_col].unique()[:amount_s]
        for student_i in students_i:
            Q_i = list(S[(S[C.student_col] == student_i) & (S[C.questions_col].isin(Q))][C.questions_col].unique())
            S_simulation = S[(S[C.student_col] != student_i) | ((S[C.student_col] == student_i) & (~S[C.questions_col].isin(Q)))]

            model_params['s_id'] = [student_i]
            model_params['Q_i'] = [Q_i]
            model_params['Q'] = [Q]

            models = generate_entity(sequencer_class, model_params)

            for question_sequencer in models:
                print "evaluating " + sequencer_class.__name__ + str(question_sequencer.params)
                start_time = time.time()

                question_sequencer.fit(S_simulation)
                fit_dur = round((time.time() - start_time) / 60.0, 3)

                model_rate = question_sequencer.rate(). \
                    sort_values(C.difficulty_col, ascending=True). \
                    drop_duplicates(C.questions_col, 'first'). \
                    sort_values(C.questions_col, ascending=True)
                rate_dur = round((time.time() - fit_dur) / 60.0, 3)

                standard_rate = S[(S[C.student_col] == student_i) & (S[C.questions_col].isin(Q_i))]. \
                    sort_values(C.difficulty_col, ascending=True). \
                    drop_duplicates(C.questions_col, 'first'). \
                    sort_values(C.questions_col, ascending=True)

                rank_result = {}
                rank_result['student_i'] = student_i
                rank_result['fit_duration'] = fit_dur
                rank_result['rate_duration'] = rate_dur
                rank_result['model_class'] = sequencer_class.__name__
                rank_result.update(question_sequencer.params)
                rank_result.update(get_rate_results(Q_i,standard_rate,model_rate))
                all_results.append(rank_result)
                print rank_result
                del question_sequencer
    pd.DataFrame(all_results).to_csv(results_path)

##FUTURE WORK
def online_evaluate(sequencer_class, model_param, amount, results_path, amount_q=5, amount_s=3,batch_size=25):
    S = get_data(amount)
    questionaires = get_questionaires('fixed', S)[:amount_q]

    all_results = []
    for questionaire in questionaires:
        print "start evaluate questionaire: " + questionaire

        Q = list(S.loc[S[C.questionaire_col] == questionaire][C.questions_col].unique())
        students_i = S[S[C.questionaire_col] == questionaire][C.student_col].unique()[:amount_s]
        for student_i in students_i:
            Q_i = list(S[(S[C.student_col] == student_i) & (S[C.questions_col].isin(Q))][C.questions_col].unique())
            S_simulation = S[(S[C.student_col] != student_i) | ((S[C.student_col] == student_i) & (~S[C.questions_col].isin(Q)))]

            model_param['s_id'] = student_i
            model_param['Q_i'] = Q_i
            model_param['Q'] = Q

            online_sequencer = sequencer_class(**model_param)

            print "evaluating " + sequencer_class.__name__ + str(online_sequencer.params)
            start_time = time.time()
            online_sequencer.fit(S_simulation)
            fit_dur = round((time.time() - start_time) / 60.0, 3)

            q_left = len(Q_i)
            S_online = pd.DataFrame(columns=[C.student_col, C.questions_col, C.difficulty_col])
            first = True
            batch_size_i = batch_size
            while q_left > 0:
                print "questions left"
                print q_left
                if q_left < batch_size_i:
                    batch_size_i = q_left

                Q_ranked_iter = model_rate = online_sequencer.rate(). \
                    sort_values(C.difficulty_col, ascending=True). \
                    drop_duplicates(C.questions_col, 'first'). \
                    sort_values(C.questions_col, ascending=True)

                Q_ranked_batch = Q_ranked_iter.sort_values(C.difficulty_col, ascending=True)[
                                 :batch_size_i]

                S_online = S_online.append(Q_ranked_batch, ignore_index=True)
                S_batch = S[(S[C.student_col] == student_i) & (
                    S[C.questions_col].isin(Q_ranked_batch[C.questions_col].unique()))]
                online_sequencer.update_model(S_batch)

                q_left -= batch_size_i

                if first:
                    first = False
                    S_offline = Q_ranked_iter.sort_values(C.difficulty_col, ascending=True)

            update_dur = round((time.time() - fit_dur) / 60.0, 3) / len(Q_i)

            s_i_q = S[(S[C.student_col] == student_i) & (S[C.questions_col].isin(Q_i))]. \
                sort_values(C.difficulty_col, ascending=True). \
                drop_duplicates(C.questions_col, 'first'). \
                sort_values(C.questions_col, ascending=True)

            S_offline = S_offline.sort_values(C.questions_col, ascending=True)
            S_offline = S_offline.sort_values(C.questions_col, ascending=True)
            rank_result = {}
            rank_result['student_i'] = student_i
            rank_result['batch_size'] = batch_size
            rank_result['question_sequencer'] = online_sequencer.__class__.__name__
            rank_result['questionaire'] = questionaire
            rank_result.update(get_rate_results(Q_i, s_i_q, S_offline, S_online))

            rank_result['duration_fit'] = fit_dur
            rank_result['duration_avg_q_update'] = update_dur
            rank_result['duration_total'] = round((time.time() - start_time) / 60.0, 3)
            all_results.append(rank_result)

            rate_dur = round((time.time() - fit_dur) / 60.0, 3)

            print rank_result
            del online_sequencer
    pd.DataFrame(all_results).to_csv(results_path)


def get_data(amount=0):
    answers_path_pp = C.answers_path_pp + str(amount) + '.csv'
    if os.path.isfile(answers_path_pp):
        S = pd.read_csv(answers_path_pp)
    else:
        if amount == 0:
            S = pd.read_table(C.answers_path)[
                     [C.student_col, 'Problem Hierarchy', C.problem_col, C.step_col, 'Correct First Attempt',
                      'Incorrects']]
        else:
            S = pd.read_table(C.answers_path)[
                    [C.student_col, 'Problem Hierarchy', C.problem_col, C.step_col, 'Correct First Attempt',
                     'Incorrects']][:amount]
        problem_df = S['Problem Hierarchy'].apply(lambda x: pd.Series([i for i in reversed(x.split(','))]))
        problem_df.rename(columns={1: 'unit', 0: 'quest'}, inplace=True)
        S = pd.concat((problem_df, S), axis=1)
        S[C.questions_col] = S[C.questionaire_col] + S[C.problem_col] + S[C.step_col]
        S = get_ranked_questions(S)
        S.to_csv(answers_path_pp)
    return S


def get_questionaires(ty,S, questionaires_to_eval=4):
    if ty == 'fixed':
        return ['Unit CTA1_10', 'Unit ES_04', 'Unit CTA1_10', 'Unit ES_07']
    questionaires = S[C.questionaire_col].unique()
    random.seed = 10
    random_questionaires = [random.randint(0, len(questionaires) - 1) for r in xrange(questionaires_to_eval)]
    random_questionaires = [questionaires[s] for s in random_questionaires]
    return random_questionaires


def main(args):

    #eval best activation function - tanh
    model_params = {
        'p_dropout': [0.25],
        'layers': [4],
        'factors': [32],
        'deep': [True],
        'batch_size': [1024],
        'activation': ['tanh', 'linear', 'relu'],
        'epochs': [1]
    }
    offline_evaluate(NCF, model_params, 250000, 'data/SVDNN_evaluations_activation_5_3.csv', 5, 3)

    # eval best factor size function - 40
    model_params = {
        'p_dropout': [0.25],
        'layers': [1],
        'factors': [80,40,20],
        'deep': [True],
        'batch_size': [1024],
        'activation': ['tanh'],
        'epochs': [1]
    }
    offline_evaluate(NCF, model_params,250000, 'data/SVDNN_evaluations_wide_5_3.csv')

    # eval layers amount size function - 40
    model_params = {
        'p_dropout': [0.25],
        'layers': [0,1,2,4,8],
        'factors': [40],
        'deep': [True],
        'batch_size': [1024],
        'activation': ['tanh'],
        'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=0, mode='auto')]],
        'epochs': [5]
    }
    offline_evaluate(NCF, model_params, 250000, 'data/SVDNN_evaluations_5_epochs_deep-small_5_3.csv')

    model_params = {}
    offline_evaluate(Edurank, model_params, 250000, 'data/Edurank_oflineEvaluations_5_3.csv')

    model_params = {
        'p_dropout': [0.25],
        'layers': [1],
        'factors': [40],
        'deep': [True],
        'batch_size': [256],
        'activation': ['tanh'],
        'callback': [[EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=0, mode='auto')]],
        'epochs': [15]
    }
    offline_evaluate(NCF, model_params, 250000, 'data/SVDNN_final.csv')


import sys
main(sys.argv[1:])