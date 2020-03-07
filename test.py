import src as th
import trueskill as ts

from importlib import reload  # Python 3.4+ only.
reload(th)
from collections import defaultdict

names = [ [1,2], [1,3], [[2],[3]] ]

history = th.History([[1,2],[1,3],[[2],[3]]],[[0,1],[1,0],[0,1]])

"Hay que agregar indices de los jugadores en las clases rearmar el camino de vuelta"
"Acá lo hago a ojo"
back = dict(history.forward)
for g in range(len(history)):#g=2
    g_names = history.games[g].names
    g_back_new_prior = [ [back[i] for i in te] if isinstance(te,list) else back[i] for te in g_names] 
    "SEGUIR: las partidas hojas usan como new prior el old prior. (tener en cuenta no hacer la corrección por old likelihood adentro)"
    for i in g_back:
        back[i] = g_back[i]


# Prueba manual de TTT
# El caso más simple 




s1_0 = th.Skill()
s2_0 = th.Skill()
s3_0 = th.Skill()

te = th.Team([s1_0,s2_0])
te.t

g = th.Game([th.Team([s1_0]),th.Team([s2_0])], [0,1])
g.t

def backpropagation(self):
    delta = 0
    history.games[-1].last_likelihood
    return delta

        
    


