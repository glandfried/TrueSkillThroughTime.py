# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
sys.path.append('../')
sys.path.append('data/')
sys.path.append('versions/')
import src as th
import mati as tm
import education as data_education
#from importlib import reload  # Python 3.4+ only.
#reload(ts)
import matplotlib.pyplot as plt
beta = 1

os.system("make")

def head_vs_branch_mati(): 
    env_th = th.TrueSkill(draw_probability=0,beta=1,tau=1 )
    env_tm = tm.TrueSkill(draw_probability=0,beta=1,tau=1 )


    prior_dict = {'e0':th.Rating(mu=20, sigma=0.001,beta=1,noise=0),
                'e1':th.Rating(mu=25,sigma=25/6,beta=1,noise=0),'e2':th.Rating(mu=29,sigma=25/6,beta=1,noise=0), 
                'e3':th.Rating(mu=31,sigma=25/6,beta=1,noise=0),'e4':th.Rating(mu=33,sigma=25/6,beta=1,noise=0)}

    history_th = env_th.History(
        data_education.composition
        , data_education.pos
        , data_education.batch_number
        , prior_dict 
        , epsilon=0.1)


    history_tm = env_tm.History(
        data_education.composition
        , data_education.pos
        , data_education.batch_number
        , prior_dict 
        , epsilon=0.1)

    history_th.through_time()
    history_tm.through_time()

    history_th.times[-1].posteriors
    history_tm.times[-1].posteriors

    res = sum(
    np.array([round(r.mu,3) for r in history_th.learning_curves['a3']]) != np.array([round(r.mu,3) for r in history_tm.learning_curves['a3']])
    ) == 0
    return res

if __name__ == "__main__":
    print("Head vs branch mati: ", head_vs_branch_mati())

