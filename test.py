import src as th
#import trueskill as ts
import numpy as np
from importlib import reload  # Python 3.4+ only.
reload(th)
from collections import defaultdict


history = th.History([[1,2],[1,3],[[2],[3]]],[[0,1],[1,0],[0,1]])
history.forward
#history.games[0][0][0].modify(th.Gaussian(0,np.inf))


def flat(xs):
    if len(xs) == 0 or not isinstance(xs[0],list):
        res = xs
    else:
        res = sum(xs,[])
    return res


back = defaultdict(lambda: th.Gaussian())
for g in reversed(range(len(history))):#g=2
    g_names = history.games[g].names
    new_prior = {}
    for n in flat(g_names):#n=3
        new_prior[n] = back[n] * history.games[g].dict_prior[n]
        """ CUANDO CAMBIO UN RATING, SE CAMBIAN TODOS
        history.games[g].dict_prior[n].modify(new_prior[n])
        history.games[g].teams[1].ratings
        """      
    
    
    history.games[g].update(new_prior)
    """
    SEGUIR
    Calcular la partida de vuelta con los new_prior!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """  
    
    for n in flat(g_names):#n=2
        back[n] = th.Skill(history.games[g].dict_likelihood[n])
            
            
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

        
    


