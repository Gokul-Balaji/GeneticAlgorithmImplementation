import unittest
import Main as mx

class TestGAFunctions(unittest.TestCase):

    def test_ChessBoard(self):
        chessBoard=mx.chessBoard(8)
        print(chessBoard)
        assert len(chessBoard)==8 and len(chessBoard[0])==8

    def test_initialPopulation(self):
        population, genotypes= mx.initialPopulation(8, 2)
        assert len(population)==2 and len(genotypes)==2 and len(genotypes[0])==8

    def test_vectortoMatrix(self):
        genotype=mx.vectortoMatrix([[0, 1, 2, 3, 4, 5, 6, 7]], 8)
        assert len(genotype)==1 and len(genotype[0])==8

    def test_fitness(self):
        fitDf=mx.fitness([[0, 1, 2, 3, 4, 5, 6, 7]],
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
                         ,8,[0, 1, 2, 3, 4, 5, 6, 7] )
        assert fitDf.shape[0]==1 and fitDf.shape[1]==9
        print(fitDf['Fitness'][0])
        assert fitDf['Fitness'][0]==56


