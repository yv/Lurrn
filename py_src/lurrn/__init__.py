from lurrn.learn import AdaGrad, FileModel, SgdMomentum, AvgMira
from lurrn.feature import FeatureHasher
from lurrn.alphabet import StringAlphabet
from lurrn.sparsmat import SparseVectorsD, SparseVectorD

__all__ = ['FeatureHasher', 'StringAlphabet',
           'SparseVectorD', 'SparseVectorsD',
           'all_learners', 'create_learner', 'load_weights']

def all_learners():
    '''
    returns a list of all learner names that can be
    used for calls to create_learner
    '''
    return ['adagrad', 'sgd_momentum', 'mira']

def create_learner(name, *args, **kwargs):
    '''
    creates a learner object
    '''
    if name == 'adagrad':
        return AdaGrad(*args, **kwargs)
    elif name == 'sgd_momentum':
        return SgdMomentum(*args, **kwargs)
    elif name == 'mira':
        return AvgMira(*args, **kwargs)
    else:
        return ValueError('no such learner')

def load_weights(fname):
    return FileModel(fname)
