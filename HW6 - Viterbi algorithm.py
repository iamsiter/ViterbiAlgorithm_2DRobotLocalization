import re
import numpy as np
import math

grid=[]
tower=[]
noisy=[]
transProb={}
dpMat=np.zeros((100,11))
path=np.zeros((100,11))

def main():
    #read data file
    read_data()
    
    for i in range(0,10):
        for j in range(0,10):
            if grid[i][j]!=0:
                moves,validMoves=getValidMoves(i,j,"out")
                transProb[(i,j)]={}
                for m in moves:
                    transProb[(i,j)][(m[0],m[1])]=1/validMoves
    viterbi()
   
    finalIndex=np.argmax(dpMat[:,-1:])
    
    printPath(finalIndex)
    
    
def printPath(finalIndex):
    trace=[]
    trace.append(finalIndex)
    for time in range(10,0,-1):
        trace.append(int(path[finalIndex][time]))
        finalIndex=int(path[finalIndex][time])
    for i in reversed(trace):
        print(getXY(i),"-->",end='')
    
def viterbi():
     #start prob
    countOnes=np.count_nonzero(grid)
    startProb=1/countOnes
    time=0
    for i in range (0,100):
        r,c=getXY(i)
        dpMat[i][time]=startProb*emissionProb(r,c,time)
        path[i][time]=i
 
    for time in range (1,11):
         for i in range(0,100):
            valid=[]
            r,c=getXY(i)
            if grid[r][c]!=0:
                moves,validMoves=getValidMoves(r,c,"in")
                prev_current_prob=[]
                for m in moves:
                    index=getFlatIndex(m[0],m[1])
                    prev_current_prob.append(((dpMat[index][time-1]*transProb[(m[0],m[1])][(r,c)]*emissionProb(r,c,time)),(m[0],m[1])))        
                
                (max_prev_current_prob,prev_move)=max(prev_current_prob)
                #check for draw cases
                tied_cases=[]
                for tie in prev_current_prob:
                    if(tie[0]==max_prev_current_prob):
                        tied_cases.append(tie[1])
                if len(tied_cases)>1:
                    prev_move=min(tied_cases)

                dpMat[i][time] = max_prev_current_prob
                path[i][time]=getFlatIndex(prev_move[0],prev_move[1])
        
def getXY(index):
    r=int(index/10)
    c=index%10;
    return r,c
        
def getFlatIndex(x,y):       
    return x*10+y       
        
def emissionProb(r,c,time):
    ep=1
    for i in range(4):
        noisyDistance=noisy[time][i];
        realDistance=findEuclideanDist(r,c,tower[i])
        
        if (noisyDistance<round(0.7*realDistance,1) or noisyDistance>round(1.3*realDistance,1)):
            return 0
        
        else:
            ep=ep*(1./(round(.6*realDistance, 1)*10+1))

    return ep
           
def findEuclideanDist(r,c,tower):
    return math.sqrt((r - tower[0]) ** 2 + (c - tower[1]) ** 2) 
        
def getValidMoves(i,j,mode):
    moves=[]
    count=0
    dx=[0,0,-1,1]
    dy=[1,-1,0,0]
    
    for k in range(4):
            if mode=="out":
                if(grid[i][j]!=0 and isValid(i+dx[k],j+dy[k])):
                    count=count+1
                    moves.append((i+dx[k],j+dy[k]))
            if mode=="in":
                if(isValid(i+dx[k],j+dy[k]) and grid[i+dx[k]][j+dy[k]]!=0):
                    count=count+1
                    moves.append((i+dx[k],j+dy[k]))
    
    return moves,count

def isValid(i,j):
    return (i>=0 and j>=0 and i<=9 and j<=9 and grid[i][j]!=0)

def read_data():
     ##Read the data file
    with open('hmm-data.txt') as file:
        lines=file.readlines();
        
    for i in range(2,12):
        grid.append([int(val) for val in lines[i].strip().split(" ")]) 
    
    for i in range(16,20):
        tower_dims=lines[i].split(":")[1].strip().split(" ")
        tower.append((int(tower_dims[0]),int(tower_dims[1])))
        
    for i in range(24,35):
        noisy_dims=re.split(" +",lines[i])
        noisy.append((float(noisy_dims[0]),float(noisy_dims[1]),float(noisy_dims[2]),float(noisy_dims[3])))
    
if __name__=="__main__":
    main()