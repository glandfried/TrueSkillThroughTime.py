# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../')
import src as ts
from importlib import reload  # Python 3.4+ only.
reload(ts)
import matplotlib.pyplot as plt
env = ts.TrueSkill(draw_probability=0,beta=1.1667,tau=(25/3)*0.1 )

plt.close()

"""

Online TTT:
    Observaciones:
        1. Las curvas de aprendizaje de online TTT y TrueSkill son muy similares
        2. La evidencia de online TTT es menor a la de trueSkill
    Objetivos:
        1. Revisar el c\'odigo.

Olvido:
    Observaciones:
        1. Al ser la curva de aprendizaje logaritmica, la tasa de olvido no debería
        ser siempre la misma. Debería ser mayor cuando en las primeras experiencias
        y deberia reducirse a medida que aumenta la experiencia
        2. Se puede poner tasa de olvido muy altas que (25%) que igual incertidumbre
        se redue bastante al haber varias partidas    
        
Grilla:
    Observaciones:
        1. Evidencia convergida conviene olvido bajo (1 sobre 8.33).
    Hip\'otesis:
        1. Porque la informaci\'on queda guardada en los vecinos
"""

beta = 1

def posiciones(a,b):
    rendimiento_alumno = np.random.normal(loc=a,scale=1)
    dificultad_examen = np.random.normal(loc=b,scale=1)
    aprueba = rendimiento_alumno >dificultad_examen 
    return [1-aprueba,0+aprueba]


def curvaDeAprendizaje(t,skill_0=15,alpha=0.2,c=0):
    return skill_0*(t**alpha)+c
def experienciaNecesearia(exp,skill_0=15,alpha=0.2):
    return (exp/skill_0)**(1/alpha)
def examen(e):
    return 25 + 5*e   


intentos = 30
universos = 6

plt.close()
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))
"""
plt.plot(experienciaNecesearia(15),15,".")
plt.plot(experienciaNecesearia(25),25,".")
plt.plot(experienciaNecesearia(30),30,".")
plt.plot(experienciaNecesearia(33.3),33.3,".")
plt.plot(experienciaNecesearia(35.5),35.5,".")
plt.plot(experienciaNecesearia(37.25),37.25,".")
"""
alumno_mues = np.full(shape=(universos,intentos+1),fill_value=None)
alumno_sigmas = np.full(shape=(universos,intentos+1),fill_value=None)
examenes_mues = np.full(shape=(universos,5),fill_value=None)
examenes_sigmas = np.full(shape=(universos,5),fill_value=None)

composition = []
pos = []
batch_number = []


u = 0
while u < universos:#u=0
    e=0; aprobados = 0; t=1
    examenes = [ts.Rating(mu=20, sigma=0.001,noise=0),
            ts.Rating(mu=25,sigma=25/6,noise=0),ts.Rating(mu=29,sigma=25/6,noise=0), 
            ts.Rating(mu=31,sigma=25/6,noise=0), ts.Rating(mu=33,sigma=25/6,noise=0)]
    while t <= intentos and e <len(examenes):#t=1
        composition.append( ['a'+str(u),'e'+str(e)])
        perf_a, perf_e = ts.Rating(curvaDeAprendizaje(t),0.0001,beta=25/6).play(), ts.Rating(examenes[e].mu,0.0001,beta=1).play() 
        pos.append(posiciones(perf_a,perf_e))
        batch_number.append(t)
        if pos[-1][0]==0:
            aprobados += 1
        else:
            aprobados = 0
        if aprobados == 3:
            e +=1; aprobados=0
        t = t + 1 
    u = u + 1
   
prior_dict = {'e0':ts.Rating(mu=20, sigma=0.0001,beta=1,noise=0),
            'e1':ts.Rating(mu=25,sigma=25/6,beta=1,noise=0),'e2':ts.Rating(mu=29,sigma=25/6,beta=1,noise=0), 
            'e3':ts.Rating(mu=31,sigma=25/6,beta=1,noise=0),'e4':ts.Rating(mu=33,sigma=25/6,beta=1,noise=0)}

reload(ts)
env_1 = ts.TrueSkill(draw_probability=0,beta=1,tau=1 )
history_1 = env_1.History(composition, pos,batch_number ,prior_dict , epsilon=0.1)
env_01 = ts.TrueSkill(draw_probability=0,beta=1,tau=0.1 )
history_01 = env_01.History(composition, pos,batch_number ,prior_dict , epsilon=0.1)

#import dill
#with open("prueba.pickle", "wb") as output_file:
#    dill.dump(history_1, output_file)

history_1.through_time()
history_1.trueSkill()
history_01.trueSkill()


np.log((10**history_01.log10_evidence_trueskill())/(10**history_1.log10_evidence_trueskill()))
np.log((10**history_01.log10_evidence_trueskill())/(10**history_1.log10_online_evidence()))
np.log((10**history_1.log10_evidence_trueskill())/(10**history_1.log10_online_evidence()))
np.log((10**history_01.log10_evidence_trueskill())/(10**history_1.log10_evidence()))

for i in history_1.learning_curves_trueskill:
    plt.plot(history_1.learning_curves_trueskill[i])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))
for i in history_1.learning_curves_online:
    plt.plot(history_1.learning_curves_online[i])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))
for i in history_1.learning_curves:
    plt.plot(history_1.learning_curves[i])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))



individual_evidence_ttt = history.individual_evidence('TTT')
individual_evidence_online = history.individual_evidence('online')
individual_evidence_trueskill = history.individual_evidence('TrueSkill')

plt.plot(history.learning_curves_online['a0'])
plt.plot(history.learning_curves_trueskill['a0'])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))

individual_evidence_ttt['a0'][0:3]
individual_evidence_online['a0'][0:3]
individual_evidence_trueskill['a0'][0:3]
plt.plot(individual_evidence_online['a0'])
plt.plot(individual_evidence_trueskill['a0'])
plt.plot(individual_evidence_ttt['a0'])



np.log10(np.prod(np.log10(individual_evidence_online['a0']))/np.prod(np.log10(individual_evidence_trueskill['a0'])))

plt.plot(np.cumsum(np.log10(ts.flat(list(map(lambda t: t.evidence, history_1.times ))))))
plt.plot(np.cumsum(np.log10(ts.flat(list(map(lambda t: t.evidence, history_01.times_trueskill ))))))

betas = np.arange(1,6)
taus =  np.arange(1,6)
evidence_ttt = np.zeros((5,5))
evidence_ts = np.zeros((5,5))
for ib in range(len(betas)):
    for it in range(len(taus )):
        env = ts.TrueSkill(draw_probability=0,beta_player=betas[ib],tau_player=taus[it] )
        history = env.History(composition, pos,batch_number ,prior_dict , epsilon=0.1)
        evidence_ttt[ib,it] = history.log10_evidence()
        evidence_ts[ib,it] = history.log10_online_evidence()

evidence_ttt[:,0]
evidence_ttt-evidence_ts


plt.plot(history.learning_curves['a0'])
plt.plot(history.learning_curves_trueskill['a0'])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))

plt.plot(history.learning_curves_at_evidence['a0'])
plt.plot(history.learning_curves_trueskill['a0'])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))

plt.plot(np.cumsum(np.log(evidence)) )
plt.plot(np.cumsum(np.log(online_evidence)) )
plt.plot(np.cumsum(np.log(evidence_ts)) )

evidence[0:5]


evidence = []
for t in history.times:
    i = 0
    for gc in t.games_composition:
        if 'a0' in ts.flat(gc):
            evidence.append(t.last_evidence[i])
        i = i +1
online_evidence = []
for t in history.times:
    i = 0
    for gc in t.games_composition:
        if 'a0' in ts.flat(gc):
            online_evidence.append(t.evidence[i])
        i = i +1
evidence_ts = [] 
for t in history.times_trueskill:
    i = 0
    for gc in t.games_composition:
        if 'a0' in ts.flat(gc):
            evidence_ts.append(t.evidence[i])
        i = i +1
            
    