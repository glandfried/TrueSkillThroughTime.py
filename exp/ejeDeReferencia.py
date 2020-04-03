# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../')
import src as ts
from importlib import reload  # Python 3.4+ only.
reload(ts)
import matplotlib.pyplot as plt
env = ts.TrueSkill(draw_probability=0,beta_player=1,tau_player=(25/3)*0.2 )
env.make_as_global()

plt.close()

###############################################
# Objetivo: 
# Aca vamos a estimar la habilidad de jugadores 
# y examenes. 

# Los examenes no tienen beta 0 porque los
# les evaluadores tienen variaciones

# Queremos ver si fijando el valor del primer 
# Examen, se puede estimar bien el resto de los
# agentes
################################################

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

u = 0
while u < universos:#u=0
    e=0; aprobados = 0; t=1
    alumno = ts.Rating(mu=25)
    alumno_mues[:,0]= alumno.mu; alumno_sigmas[:,0]= alumno.sigma
    examenes = [ts.Rating(mu=20, sigma=0.001,noise=0),
            ts.Rating(mu=25,sigma=25/6,noise=0),ts.Rating(mu=29,sigma=25/6,noise=0), 
            ts.Rating(mu=31,sigma=25/6,noise=0), ts.Rating(mu=33,sigma=25/6,noise=0)]
    while t <= intentos and e <len(examenes):#t=1
        perf_a, perf_e = ts.Rating(curvaDeAprendizaje(t),0.0001,beta=25/6).play(), ts.Rating(examenes[e].mu,0.0001,beta=1).play() 
        
        pos = posiciones(perf_a, perf_e)
        g = ts.Game( [[alumno], [examenes[e]]], pos)
        (alumno,), (examenes[e],) =  g.posterior
        #pos = posiciones(curvaDeAprendizaje(t),examenes[e+1])
        #(alumno,), (examenes[e+1],) =  ts.rate( [[alumno], [examenes[e+1]]], ranks=pos)
        alumno = alumno.forget(t=1)
        alumno_mues[u,t]= alumno.mu; alumno_sigmas[u,t]= alumno.sigma
                
        if pos[0]==0:
            aprobados += 1
        else:
            aprobados = 0
           
        if aprobados == 4:
            e +=1; aprobados=0
            alumno = ts.Rating(mu=alumno.mu,sigma=alumno.sigma)
                
        t = t + 1 
    examenes_mues[u,:]= [i.mu for i in examenes]
    examenes_sigmas[u,:]= [i.sigma for i in examenes]
    u = u + 1
   
plt.plot(np.transpose( alumno_mues))
plt.plot(curvaDeAprendizaje(np.arange(1,intentos)))

plt.plot(np.transpose( examenes_mues))
plt.plot([20,25,29,31,33])
#plt.close()

plt.plot(np.transpose( alumno_sigmas))

plt.plot(reversed(
np.mean(examenes_mues[:,4]),
np.mean(examenes_mues[:,3]),
np.mean(examenes_mues[:,2]),
np.mean(examenes_mues[:,1]),
np.mean(examenes_mues[:,0]),
))

