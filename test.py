import trueskill as th
from importlib import reload  # Python 3.4+ only.
reload(th)
from collections import defaultdict


list_teams = [ [[1,10], [3,2]] ,
               [[10,3], [2,5]] ,
               [[10,1], [2,5]] ]
list_results = [ [0,1], [1,0], [0,1]]

def intialize():
    prior = defaultdict(lambda: th.Skill(mu=25,sigma=25/3))
    
    list_likelihood = []
    list_prior = []

    n = len(list_teams)
    for g in range(n):#g=0
        teams = [ th.Team([prior[i] for i in ti ]) for ti in list_teams[g] ]
        game = th.Game(teams,list_results[0])
        list_likelihood.append(game.likelihood)
        list_prior.append(game.posterior)
        players = sum(list_teams[g], []) 
        for i in range(len(players)):
            prior[players[i]] = th.Skill(sum(list_prior[g],[])[i])
    
    return list_likelihood, list_prior, prior 
        
list_likelihood, list_prior, prior = intialize()

def backward():
    


