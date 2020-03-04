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
               [[10,3], [2,5]] ]
list_results = [ [0,1], [1,0], [0,1]]

n = len(list_teams)

history, prior = create_history()
initialize()

history[2].teams[0].performance

# Prueba manual de TTT
# El caso más simple 

reload(th)


s1_0 = th.Skill()
s2_0 = th.Skill()
s3_0 = th.Skill()

# Update
g0 = th.Game(results = [0,1], names = [[1],[2]])
g1 = th.Game([[s1_1],[s3_0]],[1,0],[[1],[3]])
[s1_2], [s3_1] = g1.posterior 
g2 = th.Game([[s2_1],[s3_1]] ,[0,1],[[2],[3]])
[s2_2], [s3_2] = g2.posterior 

# Backward
g1.update(g2.last_posterior_of(2))



history = th.History([[1,2],[1,3],[2,3]],[[0,1],[1,0],[0,1]])
len(history)


def backward():
    for g in reversed(range(n)):#g=0
        teams = [ th.Team([prior[i] for i in ti ]) for ti in list_teams[g] ]
        game = th.Game(teams,list_results[g],list_likelihood[g])
        "TODO: change Class Game()." 
        
        
    


