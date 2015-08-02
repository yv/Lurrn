import sys
import numpy
cimport numpy
from alphabet import PythonAlphabet
from lurrn.sparsmat cimport VecD1, SparseVectorD, coordinate_t
#cython: wraparound=False

cdef extern from "math.h":
    double pow(double x, double y)
    double fabs(double x)
    double sqrt(double x)
    double copysign(double x, double y)

'''
Online learners
---------------

most online learners have two important functions:
- score(vec, is_testing)
  computes a score for the respective vector. If is_testing
  is false, the learner may do some bookkeeping. If is_testing
  is true, the learner should not do that kind of bookkeeping.
- update(vecs, loss)
  updates the vector based on vecs which contains the gradient
  of one partial function, and loss (optional, defaults to 1.0)
  which contains the loss, for margin-scaling approaches
- get_weights()
  retrieves the current weight vector
- get_weights_l1()
  retrieves the l1 norm of the weight vector (may be approximate)
- set_weights(array)
  sets the current weight vector
'''

DEF soften = 1e4

cdef class AdaGrad:
    '''
    inspired by StanfordNLP's SparseAdaGradMinimizer
    but with lazy updates to weight vector
    '''
    # TODO check if this works
    cdef readonly int n_dimensions
    cdef public object fc
    cdef double eta, lambdaL1, lambdaL2
    cdef readonly double[:] weights
    cdef readonly double[:] avgWeights
    cdef double[:] sumGradSquare, lastUpdated
    cdef long int timestep
    cdef bint lazy_updates

    def __init__(self, n_dimensions,
                 l1=0.0, l2=0.0,
                 n_examples=1.0, lazy_updates=False):
        self.weights = numpy.zeros(n_dimensions, 'd')
        self.avgWeights = numpy.zeros(n_dimensions, 'd')
        self.sumGradSquare = numpy.zeros(n_dimensions, 'd')
        self.lastUpdated = numpy.zeros(n_dimensions, 'd')
        self.eta = 0.1
        self.lambdaL1 = l1 / n_examples
        self.lambdaL2 = l2 / n_examples
        self.lazy_updates = lazy_updates
    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError()
        self.weights = weights
        self.avgWeights = self.weights.copy()
    cpdef double score(self, SparseVectorD vec, bint use_avg=True):
        if use_avg:
            return vec._dotFull(& self.avgWeights[0])/self.timestep
        else:
            if self.lazy_updates:
                self.lazyUpdate(vec)
            return vec._dotFull( & self.weights[0])

    def update(self, gradient):
        cdef double a
        cdef int i
        cdef SparseVectorD vec
        cdef double gradf, prev_rate, current_rate, mix_rate, sgsValue
        cdef double wanted_update, trunc1, trunc2, actual_update
        cdef double last_timestep, idle_interval
        cdef coordinate_t feature
        for a, vec in gradient:
            for i from 0 <= i < vec.my_len:
                feature = vec.idx_ptr[i]
                gradf = a * vec.vals_ptr[i]
                prev_rate = self.eta / sqrt(
                    self.sumGradSquare[feature] + soften)
                sgsValue = self.sumGradSquare[feature] + gradf * gradf
                self.sumGradSquare[feature] = sgsValue
                current_rate = self.eta / (sqrt(sgsValue) + soften)
                wanted_update = self.weights[feature] + (current_rate * gradf)
                last_timestep = self.lastUpdated[feature]
                idle_interval = self.timestep - last_timestep - 1
                self.lastUpdated[feature] = self.timestep
                mix_rate = current_rate + prev_rate * idle_interval
                trunc1 = max(0.0, fabs(wanted_update)
                             - mix_rate * self.lambdaL1)
                trunc2 = trunc1 * pow(1 - self.lambdaL2, mix_rate)
                actual_update = copysign(trunc2, wanted_update)
                self.avgWeights[feature] += self.weights[feature] * idle_interval
                self.weights[feature] = actual_update
        self.timestep += 1
    cdef lazyUpdate(self, SparseVectorD vec):
        '''
        performs regularizer updates to weight vector that are to be
        performed lazily
        '''
        cdef int i
        cdef double current_rate, mix_rate, sgsValue
        cdef double last_timestep, idle_interval
        cdef double wanted_update, actual_update
        cdef double trunc1, trunc2
        cdef coordinate_t feature
        for i from 0 <= i < vec.my_len:
            feature = vec.idx_ptr[i]
            last_timestep = self.lastUpdated[feature]
            idle_interval = self.timestep - last_timestep - 1
            if idle_interval > 0:
                wanted_update = self.weights[feature]
                sgsValue = self.sumGradSquare[feature]
                prev_rate = self.eta / sqrt(sgsValue + soften)
                mix_rate = prev_rate * idle_interval
                trunc1 = max(0.0, fabs(wanted_update)
                             - mix_rate * self.lambdaL1)
                trunc2 = trunc1 * pow(1 - self.lambdaL2, mix_rate)
                actual_update = copysign(trunc2, wanted_update)
                self.lastUpdated[feature] = self.timestep
                self.avgWeights[feature] += self.weights[feature] * idle_interval
                self.weights[feature] = actual_update

    def __reduce__(self):
        return (make_classifier, ('adagrad',
                                  numpy.asarray(self.avgWeights)/self.timestep,
                                  self.fc))
    def save_binary(self, fname_out):
        numbers = numpy.asarray(self.avgWeights)/self.timestep
        with file(fname_out, 'w') as f:
            numpy.save(f, numbers)

cdef class SgdMomentum:
    '''
    implements stochastic gradient descent with
    momentum, and L2 regularization based on vector
    scaling (like Bottou's sgdsvm).
    Momentum makes it so that weight updates are spread over
    multiple time points.
    '''
    cdef readonly int n_dimensions
    cdef double eta
    cdef double l2
    cdef int t0
    cdef readonly double[:] weights
    cdef readonly double[:] velocity
    cdef double momentum
    cdef double weights_scale
    cdef readonly long[:] lastUpdated
    cdef long int timestep
    def __init__(self, n_dimensions, l2=0.0,
                 n_examples=1.0, momentum=0.9, eta=0.1):
        self.momentum = momentum
        self.eta = eta
        self.weights = numpy.zeros(n_dimensions, 'd')
        self.velocity = numpy.zeros(n_dimensions, 'd')
        self.lastUpdated = numpy.zeros(n_dimensions, 'l')
        self.weights_scale = 1.0
        self.n_dimensions = n_dimensions
        self.t0 = n_examples
        self.timestep = 0
        self.l2 = l2/n_examples
    cpdef score(self, SparseVectorD vec, is_testing=True):
        self.hiddenUpdates(vec)
        return self.weights_scale * vec._dotFull(& self.weights[0])
    cdef hiddenUpdates(self, SparseVectorD vec, int start=0, int end=-1):
        cdef int i, idle, k
        cdef double mom_factor
        if vec is None:
            if end == -1:
                end = self.n_dimensions
            # perform hidden updates for all parameters
            for k from start <= k < end:
                idle = self.timestep - self.lastUpdated[k]
                mom_l1 = self.momentum / (1.0 - self.l2)
                mom_factor = pow(mom_l1, idle)
                self.weights[k] += 1.0/self.weights_scale * (
                    self.velocity[k] *
                    (1.0 - mom_factor) / (1.0 - mom_l1))
                self.velocity[k] *= pow(self.momentum, idle)
                self.lastUpdated[k] = self.timestep
        else:
            for i from 0 <= i < vec.my_len:
                k = vec.idx_ptr[i]
                idle = self.timestep - self.lastUpdated[k]
                mom_l1 = self.momentum / (1.0 - self.l2)
                mom_factor = pow(mom_l1, idle)
                self.weights[k] += 1.0/self.weights_scale * (
                    self.velocity[k] *
                    (1 - mom_factor) / (1 - mom_l1))
                self.velocity[k] *= pow(self.momentum, idle)
                self.lastUpdated[k] = self.timestep
    def update(self, gradient, loss=1.0):
        cdef double a, gradf
        cdef SparseVectorD vec
        cdef coordinate_t i, k
        cdef double eta_actual
        eta_actual = self.eta * self.t0 / (self.t0 + self.timestep)
        for a, vec in gradient:
            for i from 0 <= i < vec.my_len:
                k = vec.idx_ptr[i]
                gradf = a * vec.vals_ptr[i] * eta_actual
                self.velocity[k] += gradf
        self.weights_scale *= (1.0 - eta_actual * self.l2)
        if self.weights_scale < 1e-4:
            print "SGD: rescaling, timestep=%d"%(self.timestep,)
            for k from 0<=k<self.n_dimensions:
                self.weights[k] *= self.weights_scale
            self.weights_scale = 1.0
        self.timestep += 1
    def get_weights(self):
        cdef int k
        self.hiddenUpdates(None)
        for k from 0<=k<self.n_dimensions:
            self.weights[k] *= self.weights_scale
        self.weights_scale = 1.0
        return self.weights
    def get_dense(self, start, end, testing=True):
        if not testing:
            self.hiddenUpdates(None, start, end)
        w = numpy.zeros(end-start)
        w[:] = self.weights[start:end]
        w *= self.weights_scale
        return w
    def get_weights_l1(self):
        cdef double result = 0.0
        cdef int k
        for k from 0<=k<self.n_dimensions:
            result += fabs(self.weights[k])
        return result * self.weights_scale
    def set_weights(self, w):
        self.weights[:] = w

    def save_binary(self, fname_out):
        numbers = self.get_weights()
        with file(fname_out, 'w') as f:
            numpy.save(f, numbers)

cdef class AvgPer:
    '''
    Simple averaged perceptron
    '''
    # TODO check if this works
    cdef readonly int n_dimensions
    cdef public object fc
    cdef double eta
    cdef readonly double[:] weights
    cdef readonly double[:] avgWeights
    cdef double[:] lastUpdated
    cdef long int timestep

    def __init__(self, n_dimensions):
        self.weights = numpy.zeros(n_dimensions, 'd')
        self.avgWeights = numpy.zeros(n_dimensions, 'd')
        self.lastUpdated = numpy.zeros(n_dimensions, 'd')
        self.timestep = 0
    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError()
        self.weights = weights
        self.avgWeights = self.weights.copy()
    cpdef double score(self, SparseVectorD vec, bint use_avg=True):
        if use_avg and self.timestep > 0:
            return vec._dotFull(& self.avgWeights[0])/self.timestep
        else:
            return vec._dotFull( & self.weights[0])

    def update(self, gradient):
        cdef double a
        cdef int i
        cdef SparseVectorD vec
        cdef double gradf
        cdef double actual_update
        cdef double last_timestep, idle_interval
        cdef unsigned long feature
        for a, vec in gradient:
            for i from 0 <= i < vec.my_len:
                feature = vec.idx_ptr[i]
                gradf = a * vec.vals_ptr[i]
                last_timestep = self.lastUpdated[feature]
                idle_interval = self.timestep - last_timestep
                self.lastUpdated[feature] = self.timestep
                actual_update = self.weights[feature] + gradf
                self.avgWeights[feature] += self.weights[feature] * idle_interval
                self.weights[feature] = actual_update
        self.timestep += 1

    def __reduce__(self):
        return (make_classifier, ('avgper',
                                  numpy.asarray(self.avgWeights)/self.timestep,
                                  self.fc))
    def save_binary(self, fname_out):
        numbers = numpy.asarray(self.avgWeights)/self.timestep
        with file(fname_out, 'w') as f:
            numpy.save(f, numbers)

cdef class AvgMira:
    '''
    Simple averaged MIRA
    '''
    # TODO check if this works
    cdef readonly int n_dimensions
    cdef public object fc
    cdef double C
    cdef readonly double[:] weights
    cdef readonly double[:] avgWeights
    cdef double[:] lastUpdated
    cdef long int timestep

    def __init__(self, n_dimensions, C=0.1, l1=0.0, l2=0.0,
                 n_examples=None):
        self.weights = numpy.zeros(n_dimensions, 'd')
        self.avgWeights = numpy.zeros(n_dimensions, 'd')
        self.lastUpdated = numpy.zeros(n_dimensions, 'd')
        self.timestep = 0
        self.C = C
        if l1 != 0.0:
            print >>sys.stderr, "WARNING: AvgMira does not support l1 reg"
        if l2 != 0.0:
            print >>sys.stderr, "WARNING: AvgMira does not support l2 reg"

    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError()
        self.weights = weights
        self.avgWeights = self.weights.copy()
    def get_weights_l1(self):
        cdef double result = 0.0
        cdef int k
        for k from 0<=k<self.n_dimensions:
            result += fabs(self.avgWeights[k])
        return result / <double>self.timestep
    cpdef double score(self, SparseVectorD vec, bint use_avg=True):
        if use_avg and self.timestep > 0:
            self.hiddenUpdates(vec)
            return vec._dotFull(& self.avgWeights[0])/self.timestep
        else:
            return vec._dotFull( & self.weights[0])

    cdef hiddenUpdates(self, SparseVectorD vec, int start=0, int end=-1):
        cdef int i
        cdef unsigned long feature
        if vec is not None:
            for i from 0<= i < vec.my_len:
                feature = vec.idx_ptr[i]
                last_timestep = self.lastUpdated[feature]
                idle_interval = self.timestep - last_timestep
                if idle_interval > 0:
                    self.avgWeights[feature] += self.weights[feature] * idle_interval
                    self.lastUpdated[feature] = self.timestep
        else:
            if end == -1:
                end = self.n_dimensions
            for feature from start<= feature < end:
                last_timestep = self.lastUpdated[feature]
                idle_interval = self.timestep - last_timestep
                if idle_interval > 0:
                    self.avgWeights[feature] += self.weights[feature] * idle_interval
                    self.lastUpdated[feature] = self.timestep

    def update(self, gradient, double loss=1.0):
        cdef double a
        cdef int i
        cdef SparseVectorD vec
        cdef double margin, norm, alpha
        cdef double gradf
        cdef double actual_update
        cdef double last_timestep, idle_interval
        cdef unsigned long feature
        # Step 1: calculate update size (alpha s.t. gradient*w >= loss)
        # <w+grad*alpha, grad> = <w,grad> + alpha*||grad||^2
        # d.h. <w,grad> + alpha * ||grad||^2 >= loss
        # d.h. alpha * ||grad||^2 >= loss - <w,grad>
        # d.h. alpha >= (loss - <w,grad>) / ||grad||^2
        vec = gradient.to_vec()
        margin = loss - self.score(vec, False)
        assert margin >= 0, margin
        norm = vec.dotSelf()
        alpha = margin / norm
        # Step 2: actual update
        for i from 0 <= i < vec.my_len:
            feature = vec.idx_ptr[i]
            gradf = alpha * vec.vals_ptr[i]
            last_timestep = self.lastUpdated[feature]
            idle_interval = self.timestep - last_timestep
            self.lastUpdated[feature] = self.timestep
            actual_update = self.weights[feature] + gradf
            self.avgWeights[feature] += self.weights[feature] * idle_interval
            self.weights[feature] = actual_update
        self.timestep += 1
    def get_dense(self, start, end, testing=True):
        if not testing:
            self.hiddenUpdates(None, start, end)
            w = numpy.zeros(end-start)
            w[:] = self.weights[start:end]
        else:
            w = numpy.zeros(end-start)
            w[:] = self.avgWeights[start:end]
            w *= 1.0/self.timestep
        return w

    def __reduce__(self):
        return (make_classifier, ('mira',
                                  numpy.asarray(self.avgWeights)/self.timestep,
                                  self.fc))
    def save_binary(self, fname_out):
        numbers = numpy.asarray(self.avgWeights)/self.timestep
        with file(fname_out, 'w') as f:
            numpy.save(f, numbers)
# The MIRA implementation in Moses has the following additional stuff:
# - averaging only over current epoch (instead of all seen weight vectors)
# - Moses uses C=0.01 as default
# - Moses has a learning rate (def. 1) for vector updates

cdef class FileModel:
    cdef readonly int n_dimensions
    cdef public object fc
    cdef readonly double[:] weights
    def __init__(self, fname):
        numbers = numpy.load(fname)
        self.weights = numbers
        self.n_dimensions = len(numbers)
    cpdef double score(self, SparseVectorD vec, bint use_avg=True):
        return vec._dotFull(& self.weights[0])
    def get_dense(self, start, end):
        w = numpy.zeros(end-start)
        w[:] = self.weights[start:end]
        return w

def make_classifier(tp, weights, fc=None):
    if tp == 'adagrad':
        result = AdaGrad(len(weights))
        result.set_weights(weights)
        result.fc = fc
        return result
    if tp == 'avgper':
        result = AvgPer(len(weights))
        result.set_weights(weights)
        result.fc = fc
        return result
