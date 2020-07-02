import numpy
import pandas as pd
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Function to Select Parents
def selectParent(fitnessMatrix,populationSize):
    fitnessMatrix=fitnessMatrix.sort_values(by='Rank')
    #Selecting the Fittest Parent for CrossOver and Mutation
    fittestParentPopulation=pd.DataFrame
    #fittestParentFitness=[]
    fittestParentPopulation=fitnessMatrix.iloc[:populationSize,:]
    #print(fittestParentPopulation)
    #crossOver(fittestParentPopulation)
    return fittestParentPopulation

#Function to Mutate the Offsprings
def mutation(crossedParents, mutationRate,queens):
    mutatedParents=[]
    #print("Mutated:{}".format(len(crossedParents)))
    toMutate=math.ceil(mutationRate*queens*len(crossedParents))
    for times in range(toMutate):
        randomchild=numpy.random.random_integers(0,len(crossedParents)-1).astype(int)
        mutationPositon=numpy.random.random_integers(0,len(crossedParents[0])-1).astype(int)
        initialMutationRow=numpy.random.random_integers(0,len(crossedParents[0])-1).astype(int)
        '''
        if parent.count(initialMutationRow)==0:
            parent[mutationPositon]=initialMutationRow
        else:
            while parent.count(initialMutationRow)!=0: #initialMutationRow!=parent[mutationPositon]:
                initialMutationRow = numpy.random.random_integers(0, len(parent) - 1).astype(int)
        '''
        crossedParents[randomchild][mutationPositon] = initialMutationRow
        mutatedOffsprings=crossedParents
        #parent[mutationPositon] = initialMutationRow
        #print("{}. Mutated {} to {}".format(times, crossedParents[randomchild],mutatedOffsprings[randomchild]))
    #print(mutatedParents)
    return mutatedOffsprings

#Function to Calculate the Fitness
def fitnessCheck(df,fitnessValue):
    if df['Solution'][0]==fitnessValue:
        print('Right Sequence Generated...')
        #print(df['Solution'][0])
        #print(df.iloc[:1,:])
        return True
    else:
        return False

#Function to Reproduce and Generate offsprings
def crossOver(parents, crossrate,crossoverPopulation, parentsToKeep, mutationRate, queens):
    eliteParents=[]
    offspringsAfterCross=[]
    parentsLeft=[]
    nextGenIndividuals=[]
    #Elite Individuals
    for parent in range(parentsToKeep): #Parents to move to Next Generation
        eliteParents.append(parents['Solution'].values.tolist()[parent])
    #print("Elite:{}".format(len(parentsAfterCross)))
    #CrossOver Individuals
    for child in range(parentsToKeep+1, crossoverPopulation+1): #len(parents['Solution'].values.tolist()
        if len(parents['Solution'].values.tolist())>child:
            firstChild = parents['Solution'].values.tolist()[child][0:crossrate] + \
                         parents['Solution'].values.tolist()[child + 1][crossrate:]
            secondChild = parents['Solution'].values.tolist()[child + 1][0:crossrate] + \
                          parents['Solution'].values.tolist()[child][crossrate:]
        else:
            firstChild = parents['Solution'].values.tolist()[child][0:crossrate] \
                         + parents['Solution'].values.tolist()[-1][crossrate:]
            secondChild = parents['Solution'].values.tolist()[-1][0:crossrate] + \
                          parents['Solution'].values.tolist()[child][crossrate:]

        #parentsAfterCross.append(parents['Solution'].values.tolist()[0])
        #parentsAfterCross.append(parents['Solution'].values.tolist()[1])
        offspringsAfterCross.append(firstChild)
        offspringsAfterCross.append(secondChild)

    parentsAfterCrossMutation=mutation(offspringsAfterCross,mutationRate,queens)
    for child in range(crossoverPopulation+1,len(parents['Solution'].values.tolist())):
        parentsLeft.append(parents['Solution'].values.tolist()[child])

    nextGenIndividuals=eliteParents+parentsAfterCrossMutation+parentsLeft
    #print("Elite: {}, CrossMutated: {}, Left to Survive: {}".format(len(eliteParents), len(parentsAfterCrossMutation), len(parentsLeft)))
    #print("Elite: {}".format(eliteParents))
    #print("Crossed & Mutated:{}".format(len(parentsAfterCrossMutation)))
    #print("Parents Left to Survive:{}".format(len(parentsLeft)))

    return nextGenIndividuals





