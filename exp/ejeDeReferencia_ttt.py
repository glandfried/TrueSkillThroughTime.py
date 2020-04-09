# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../')
import src as ts
from importlib import reload  # Python 3.4+ only.
reload(ts)
import matplotlib.pyplot as plt
env = ts.TrueSkill(draw_probability=0,beta_player=4.1667,tau_player=(25/3)*0.2 )

plt.close()

"""
Conlusiones:    
    Olvido:
        1. Al ser la curva de aprendizaje logaritmica, la tasa de olvido no debería
        ser siempre la misma. Debería ser mayor cuando en las primeras experiencias
        y deberia reducirse a medida que aumenta la experiencia
        2. Se puede poner tasa de olvido muy altas que (25%) que igual incertidumbre
        se redue bastante al haber varias partidas    
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


intentos = 70
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
        if aprobados == 4:
            e +=1; aprobados=0
        t = t + 1 
    u = u + 1
   
prior_dict = {'e0':ts.Rating(mu=20, sigma=0.0001,beta=1,noise=0),
            'e1':ts.Rating(mu=25,sigma=25/6,beta=1,noise=0),'e2':ts.Rating(mu=29,sigma=25/6,beta=1,noise=0), 
            'e3':ts.Rating(mu=31,sigma=25/6,beta=1,noise=0),'e4':ts.Rating(mu=33,sigma=25/6,beta=1,noise=0)}
reload(ts)
history = env.History(composition, pos,batch_number ,prior_dict , epsilon=0.1)
history.log10_evidence()
history.log10_evidence_trueskill()

for i in history.learning_curves_trueskill:
    plt.plot(history.learning_curves_trueskill[i])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))
for i in history.learning_curves:
    plt.plot(history.learning_curves[i])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))
for i in history.learning_curves_at_evidence:
    plt.plot(history.learning_curves_at_evidence[i])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))

plt.plot(history.learning_curves_trueskill['a0'][0:5])
plt.plot(history.learning_curves_at_evidence['a0'][0:5])
plt.plot(history.learning_curves['a0'][0:5])
plt.plot(curvaDeAprendizaje(np.arange(1,intentos))[0:5])
plt.plot(evidence[0:5] )
plt.plot(evidence_ts[0:5] )
evidence = []
for t in history.times:
    i = 0
    for gc in t.games_composition:
        if 'a0' in ts.flat(gc):
            evidence.append(t.evidence[i])
        i = i +1
evidence_ts = [] 
for t in history.times_trueskill:
    i = 0
    for gc in t.games_composition:
        if 'a0' in ts.flat(gc):
            evidence_ts.append(t.evidence[i])
        i = i +1
            
    