Lurrn
=====

Lurrn is a library for online learning of linear classifiers which can be
used to learn, e.g. CRFs. It's mostly meant as an educational tool for
demonstrating structured learning, but could also be used for more serious
undertakings.

Basic usage
-----------

For a medium complexity example, see examples/linear_crf.py.

The basic idea is: you represent your example as a list of strings,
use a FeatureHasher to turn this into a (sparse) feature vector,
and can then hand it to a learner to get its score wrt. a weight
vector (learner.score(vec)), or to update the weight vector with
this gradient (learner.update([(1.0, vec)]). Using the SparseVectorsD
class, you can also combine multiple vectors for a gradient update.
