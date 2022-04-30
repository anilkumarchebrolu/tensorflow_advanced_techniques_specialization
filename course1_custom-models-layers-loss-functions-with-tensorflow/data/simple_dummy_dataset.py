import numpy as np

class SimpleDummyDataSet:
    def __init__(self) -> None:
        pass
    
    def simply_dummy_dataset(self):
        '''
        Here Our dummy dataset is just a pair of arrays xs and ys defined by the relationship  𝑦=2𝑥−1 . 
        xs are the inputs while ys are the labels.
        '''
        # inputs
        X = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

        # labels
        Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
        return X, Y