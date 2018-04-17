import numpy as np
from hmmlearn import hmm

def emissionProba(i,time):
        noisyDistance=obs[time]
        realDistance=i+1

        if (noisyDistance<round(0.7*realDistance,1) or noisyDistance>round(1.3*realDistance,1)):
            return 0
        
        else:
            ep=(1./(round(.6*realDistance, 1)*10+1))
        return ep
    
model = hmm.MultinomialHMM(n_components=6)

obs=np.atleast_2d([5.5,3.0,3.6,4.0,4.5,5.4]).T

model.startprob_ = np.array([0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667]).T

model.transmat_ =np.array([[0,1,0,0,0,0],[0.5,0,0.5,0,0,0],[0,0.5,0,0.5,0,0],[0,0,0.5,0,0.5,0],[0,0,0,0.5,0,0.5],[0,0,0,0,1,0]])

emitmat=np.zeros((6,6),dtype=float)

for i in range(6):
       for j in range(6):
                   emitmat[i][j]=emissionProba(i,j)  

model.emissionprob_ = emitmat
                
# Predict the optimal sequence of internal hidden state
X = np.atleast_2d([5,3,3,4,2,5]).T
print(model.decode(X,algorithm="viterbi"))
