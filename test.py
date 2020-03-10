import src as th
#import trueskill as ts

from importlib import reload  # Python 3.4+ only.
reload(th)
#from collections import defaultdict

names = [ [1,2], [1,3], [[2],[3]] ]

history = th.History([[1,2],[1,3],[[2],[3]]],[[0,1],[1,0],[0,1]])

for i in range(3):
    res = dict(zip(sum(history.games[i].names,[]) if isinstance(history.games[i].names[0],list) else names[i],
         sum([te.ratings for te in history.games[i].teams ],[])))
dict(zip(names[1],sum([te.ratings for te in history.games[1].teams ],[])))


history.games[2].names

for t in history.games[2].teams:
    print(t.ratings)

history.games[2].ratings
history.games[1].ratings
history.games[0].ratings

history.games[2].names

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

        
    


