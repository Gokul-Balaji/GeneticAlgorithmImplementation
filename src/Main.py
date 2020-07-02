import numpy
import pandas as pd
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import GeneticAlgorithm as GA
#import matplotlib.pyplot as plt
#import seaborn as sns
#import unittest


def test_ChessBoard():
    chessboard = chessBoard(8)
    #print(chessboard)
    assert len(chessboard) == 8 and len(chessboard[0]) == 8

def test_initialPopulation():
    population, genotypes = initialPopulation(8, 2)
    assert len(population) == 2 and len(genotypes) == 2 and len(genotypes[0]) == 8

def test_vectortoMatrix():
    genotype = vectortoMatrix([[0, 1, 2, 3, 4, 5, 6, 7]], 8)
    assert len(genotype) == 1 and len(genotype[0]) == 8

def test_fitness():
    fitDf = fitness([[0, 1, 2, 3, 4, 5, 6, 7]],
                       [[[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]]],
                       [['0,0', '0,1', '0,2', '0,3', '0,4', '0,5', '0,6', '0,7'],
                        ['1,0', '1,1', '1,2', '1,3', '1,4', '1,5', '1,6', '1,7'],
                        ['2,0', '2,1', '2,2', '2,3', '2,4', '2,5', '2,6', '2,7'],
                        ['3,0', '3,1', '3,2', '3,3', '3,4', '3,5', '3,6', '3,7'],
                        ['4,0', '4,1', '4,2', '4,3', '4,4', '4,5', '4,6', '4,7'],
                        ['5,0', '5,1', '5,2', '5,3', '5,4', '5,5', '5,6', '5,7'],
                        ['6,0', '6,1', '6,2', '6,3', '6,4', '6,5', '6,6', '6,7'],
                        ['7,0', '7,1', '7,2', '7,3', '7,4', '7,5', '7,6', '7,7']]
                       , 8, [0, 1, 2, 3, 4, 5, 6, 7])
    assert fitDf.shape[0] == 1 and fitDf.shape[1] == 9
    #print(fitDf['Fitness'][0])
    assert fitDf['Fitness'][0] == 56

#Define the Chess Board with the Number of Queens
def chessBoard(size):
    chessBoard=[]
    for rows in range(0, size):
        #print("Row : {}".format(rows))
        chessColumns=[]
        for columns in range(0,size):
            #print("Column : {}".format(columns))
            #print("Position : ({},{})".format(rows,columns))
            chessColumns.append(str(str(rows)+","+str(columns)))
        chessBoard.insert(rows,chessColumns)
    return chessBoard



#Define Initial Population
def initialPopulation(chessBoard, solutions):
    population=[]
    for colPosition in range(0,solutions):
        #print(colPosition)
        populationIndices=numpy.random.rand(chessBoard)*chessBoard
        populationIndices=populationIndices.astype(numpy.uint8).flatten()
        population.insert(colPosition,populationIndices.tolist())
        #print(populationIndices)
    #print(population)
    genotype=vectortoMatrix(population,chessBoard)
    return population, genotype

#Convert Phenotype to Genotype
def vectortoMatrix(population, chessboard):
    genotype=[]
    for genepool in population:
        gene=[]
        for phenotype in range(len(genepool)):
            #print(genepool[phenotype])
            initialGene = numpy.zeros(chessboard).astype(numpy.uint8).tolist()
            #print(initialGene)
            initialGene[genepool[phenotype]]=1
            #print(initialGene)
            gene.append(initialGene)
        genotype.append(gene)
        #print(gene)
        #print(genotype)
    return genotype

#Safe Divide Fitness to Probability
def safeDivision(x,y):

    if y==0:
        return 1
    else:
        return x/y

#Calculate Fitness of the Solution
def fitness(population,genepool, chessBoard, queens,fitnessValue):
    #print()
    fitnessValue=fitnessValue
    rowAttacksCount, queenPositions=rowAttacks(population, genepool, chessBoard)
    diagonalAttacksCount=diagAttacks(population, genepool, chessBoard)
    #print(rowAttacksCount,diagonalAttacksCount)
    #zipped=zip(rowAttacksCount, diagonalAttacksCount)
    #print(zipped)
    #fitness=[list(map(add, rowAttacksCount, diagonalAttacksCount))]
    #fitness=map(sum, zip(rowAttacksCount, diagonalAttacksCount))
    fitness = [(row + diag) for row, diag in zip(rowAttacksCount, diagonalAttacksCount)]
    configScore=calculateConfig(population, genepool, queens,fitnessValue)
    probability=[round(safeDivision(1,1+(row+diag)),4) for row, diag in zip(rowAttacksCount, diagonalAttacksCount)]
    #total=probability.sum()
    totalProb=[gene/sum(probability) for gene in probability]
    #probability=[1-round(1/(row+diag),4) for row, diag in zip(rowAttacksCount, diagonalAttacksCount)]
    #print("Probability of Initial Population")
    #print(probability)
    #print("Probablity of Initial Population")
    #print(probability)
    queensFitnessZip= list(zip(population, queenPositions,rowAttacksCount,diagonalAttacksCount,fitness,probability, totalProb, configScore))
    queenFitnessDf=pd.DataFrame(queensFitnessZip, columns=['Solution', 'Queens', 'RowAttack', 'DiagonalAttack', 'Fitness','Fitness Prob','Probability', 'Configuration Score'])
    queenFitnessDf['Rank']=queenFitnessDf['Configuration Score'].rank(ascending=False, method='dense').astype(int).sort_values(ascending=True)
    #print(queenFitnessDf.sort_values(by='Rank'))
    return queenFitnessDf.sort_values(by='Rank')

#Calculate the Config Function
def calculateConfig(population, genepool, queens, expectedValue):
    configScore=[]
    for solution in range(len(population)):
        score=0
        for gene in range(len(population[solution])):
            if population[solution][gene]==expectedValue[gene]:
                score+=1
        configScore.append((score/queens)*100)
    return configScore


#Calculate the Number of Diagonal Attacks
def diagAttacks(population, genepool, chessBoard):

    diagonalAttacks=[]
    for gene in genepool:
        queens = []
        for chrome in range(len(gene)):
            queen, row = queenPosition(gene[chrome], chrome)
            queens.append(queen)
        diagonalAttacks.append(calculateDiagAttacks(queens))
    #print("Attacks for Queen in Diagonal")
    #print(diagonalAttacks)
    return diagonalAttacks

def calculateDiagAttacks(queens):
    #print(queens)
    attacksinDiagonal=0
    attacksinDiagonalList=[]
    #print("Attacks for Queen in Diagonal")
    for queen in queens:
        qrow, qcol = queen.split(",")
        attacksinDiagonal = 0
        #print("Queen")
        #print(qrow, qcol)
        for attackingQueen in queens:
            aqrow, aqcol = attackingQueen.split(",")
            #print("Attacking Queen")
            #print(attackingQueen,aqrow, aqcol)
            if qrow!=aqrow and qcol!=aqcol:
                if abs(int(qrow)-int(aqrow))==abs(int(qcol)-int(aqcol)):
                    attacksinDiagonal+=1
            #print(attacksinDiagonal)
        attacksinDiagonalList.append(attacksinDiagonal)
    #print(attacksinDiagonalList)
    return sum(attacksinDiagonalList)

#Calculate the Number of Row Attacks
def rowAttacks(population, genepool, chessBoard):
    Position=[]
    queenRowAttackPool = []
    for genes in genepool: # [ [ ], [ ] ]
        queenPositioninGene = []
        rowPositioninGene=[]
        for chromosome in range(len(genes)): # [ ]
            #print(chromosome)
            #print(genes)
            queenPositioninChromosome, rowPositioninChromosome=queenPosition(genes[chromosome], chromosome)
            queenPositioninGene.append(queenPositioninChromosome)
            rowPositioninGene.append(rowPositioninChromosome)
        #print(queenPositioninGene)
        queenRowAttackPool.append(attacksinRow(rowPositioninGene))
        Position.append(queenPositioninGene)

    #print("Position of Queens")
    #print(Position) #Queen Position
    #print("Attacks for Queen in Rows")
    #print(queenRowAttackPool) #Row Attacks in Gene Pool
    return queenRowAttackPool, Position

def attacksinRow(queensRowPosition):
    queenAttack=[]
    rowAttack=0
    for queen in range(len(queensRowPosition)):
        attack=0
        for attackingQueens in range(len(queensRowPosition)):
            if queen!=attackingQueens and queensRowPosition[queen]==queensRowPosition[attackingQueens]:
                attack+=1
        queenAttack.append(attack)
    for attack in queenAttack:
        rowAttack=rowAttack+attack
    return rowAttack

#Identify the Queen Positions
def queenPosition(chromosome, columnNumber):
    #queenPosition=[]
    for chrom in range(len(chromosome)):
        if chromosome[chrom] == 1:
            rowPosition = chrom  # Gets me the Position of a Queen in a Column x Row
    queenPosition=(str(rowPosition)+","+str(columnNumber)) #Positions of Queens
    rowPosition=rowPosition #Positions of Queens
    #print(queenPosition)
    return queenPosition, rowPosition


if __name__ == '__main__':
    #testRunner=unitTesting.main()
    chessBoard
    '''print("Enter the Number of Queens :")
    queens=int(input())'''
    queens=8 #Number of Queens on the Board
    chessboard=chessBoard(queens)
    #print("Input the CrossOver Probability [0-1]:")
    #crossOverInput=float(input())
    '''
    while True:
        if crossOverInput>=0.0 and crossOverInput<=1.0:
            print("Valid CrossOver Probability...")
            break
        else:
            print("Invalid CrossOver Probability. Please re-enter right CrossOver Probability.")
            crossOverInput = float(input())
    '''
    #crossOverRate=crossOverInput
    crossOverRate=0.6 # CrossOver Rate - Probability of Crossing Parents [CrossRate*Genes]
    crossOverpoint=2
    fitnessValue = [6, 2, 7, 1, 4, 0, 5, 3]
    #print("Input the Mutation Probability [0-1]:")
    #muteInput=float(input())
    '''while True:
        if muteInput>=0.0 and muteInput<=1.0:
            #print("Valid Mutation Probability...")
            break
        else:
            #print("Invalid Mutation Probability. Please re-enter right Mutation Probability.")
            muteInput = float(input())'''
    #mutationRate=muteInput
    mutationRate=0.1 #Probability of Mutating the Child Offsprings per Generation [MutationRate*Population in Generation]
    generations=10000 #Number of Generations
    parenttoKeep=(0.05) #Number of Parents to keep in a Generation
    #print("Please Input the Population Size.")
    #populationSize=int(input())
    #print("Creating Initial Population...")
    populationSize=100 #Initial Population
    '''gen = []
    fit = []
    fig, ax = plt.subplots()'''
    # plt.plot(generation, fitnessDataFrame['Fitness'][0])
    #plt.xlabel("Generations")
    #plt.ylabel("Fitness")
    label = "PS:" + str(populationSize) + ";Cp:" + str(crossOverRate) + ";Mp:" + str(mutationRate)
#print(chessBoard)
    population, genepool=initialPopulation(queens, populationSize)
    #print(population)
    fitnessDataFrame=fitness(population,genepool,chessboard, queens,fitnessValue)
    #Tabular View of Initial Population with Fitness Score

    print("Population Size:{} ".format(populationSize))
    print("Elite Individuals:{} ".format(round(parenttoKeep*populationSize)))
    print("CrossOver Probability:{}".format(crossOverRate))
    print("Mutation Probability:{} ".format(mutationRate))
    print("Initial Population with Fitness Scores & Rank")

    print(fitnessDataFrame)
    test_ChessBoard()
    test_initialPopulation()
    test_vectortoMatrix()
    test_fitness()
    #plt.text(60, .025, r'$\Pc=100,\ \Pm=15$')
    print("Genetic Algorithm Implementation")
    for generation in range(generations):
        print("Generation: {} {}".format(generation,fitnessDataFrame.iloc[:1,:].to_string(header=False, index=False)))
        #gen.append(generation)
        #fit.append(fitnessDataFrame['Fitness'][0])
        #print("Generation Population:{}".format(len(fitnessDataFrame['Solution'].values.tolist())))
        #print(fitnessDataFrame.iloc[:populationSize,:].to_string(header=False))
        if GA.fitnessCheck(fitnessDataFrame,fitnessValue)==True:
            print("Optimal Solution Achieved at Generation: {}".format(generation))
            print(fitnessDataFrame['Solution'][0])
            #print(fitnessDataFrame)
            #plt.show()
            break
        if generation==generations-1 and GA.fitnessCheck(fitnessDataFrame, fitnessValue)==False:
            print("Optimal Solution Not-Achieved")
            print(fitnessDataFrame['Solution'][0])

        fittestParents=GA.selectParent(fitnessDataFrame,populationSize) #Sorted based on Rank/Fitness Value
        nextGenPopulation=GA.crossOver(fittestParents, crossOverpoint, (math.ceil(crossOverRate*populationSize)),
                                       math.ceil(parenttoKeep*populationSize), mutationRate, queens)
        #print("Next Gen Population:{}".format(len(crossedMutatedPopulation)))
        #mutatedPopulation=GA.mutation(crossedPopulation,int(mutationRate*queens*populationSize))
        genePoolAfter=vectortoMatrix(nextGenPopulation, queens)
        #print(genePoolAfter)
        fitnessDataFrame=fitness(nextGenPopulation,genePoolAfter,chessboard, queens,fitnessValue)
        #print(fitnessDataFrame.iloc[:1,:])
        #print(fitnessDataFrame)
    #plt.plot(gen, fit, label=label)
    #plt.set()
    #ax.plot(gen, fit, label=label)
    #ax.set(xlabel='Generations', ylabel='Fitness',label=label)
    #ax.set_xlim(0,max(gen))
    #ax.set_ylim(0,max(fit))
    #ax.set_title(label)
    #ax.grid()
    #ax.legend()
    #fig.savefig("D:\Gokul Balaji\Analytics World\GeneticAlgorithm\GA\Result7.png")
    #plt.show()
    #plt.show()

    #sns.lineplot("Generation","Fitness", data=pd.DataFrame[gen,fit],color="coral", label=label,markers='o' )


