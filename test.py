import trueskill as th
from importlib import reload  # Python 3.4+ only.
reload(th)
from collections import defaultdict

def initialize():
    "First approach"
    prior = defaultdict(lambda: th.Skill(mu=25,sigma=25/3))
    history = []
    list_likelihood = []
    list_prior = []

    for g in range(n):#g=0
        teams = [ th.Team([prior[i] for i in ti ]) for ti in list_teams[g] ]
        game = th.Game(teams,list_results[g])
        history.append(game)
        list_likelihood.append(game.likelihood)
        list_prior.append(game.posterior)
        players = sum(list_teams[g], []) 
        for i in range(len(players)):
            prior[players[i]] = th.Skill(sum(list_prior[g],[])[i])
    
    return list_likelihood, list_prior, prior , history
        
def update_prior(players,posterior,prior):
    for i in range(len(players)):
        prior[players[i]] = th.Skill(sum(posterior,[])[i])
        

def create_history():
    "Second approach"
    prior = defaultdict(lambda: th.Skill(mu=25,sigma=25/3))
    history = []
    for g in range(n):#g=0
        teams = [ th.Team([prior[i] for i in ti ]) for ti in list_teams[g] ]
        history.append(th.Game(teams,list_results[g]))
        update_prior(sum(list_teams[g],[]) ,history[-1].posterior,prior)
    return history, prior
    

list_teams = [ [[1,10], [3,2]] ,
               [[10,3], [2,5]] ,
               [[10,1], [2,5]] ]
list_results = [ [0,1], [1,0], [0,1]]

n = len(list_teams)

history, prior = create_history()
initialize()

history[2].teams[0].performance


def backward():
    for g in reversed(range(n)):#g=0
        teams = [ th.Team([prior[i] for i in ti ]) for ti in list_teams[g] ]
        game = th.Game(teams,list_results[g],list_likelihood[g])
        "TODO: change Class Game()." 
        
        
    


