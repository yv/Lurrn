# -*- encoding: utf-8 -*-
import re
import sys
import optparse
import numpy
from shutil import copyfile
from itertools import izip
from time import time
from random import shuffle
from lurrn import FeatureHasher, create_learner, load_weights, \
    CPPAlphabet, SparseVectorsD, SparseVectorD
from lurrn.item_logging import set_logfile_prefix, LongLogEntry, add_log_item

tag_alphabet = CPPAlphabet()


def read_tabular(fname, columns_idx):
    '''
    reads a file with tabular data and extracts the
    given columns from each sentence.
    '''
    columns = [[] for x in columns_idx]
    with file(fname) as f_in:
        for l in f_in:
            line = l.strip().split()
            if line == [''] or line == []:
                if columns[0]:
                    yield columns
                    columns = [[] for x in columns_idx]
            else:
                for i, col in enumerate(columns_idx):
                    try:
                        columns[i].append(line[col])
                    except IndexError:
                        columns[i].append('_')
        if columns[0]:
            yield columns

class LinearCRF:
    '''
    simple Bigram CRF with unigram features
    '''

    def __init__(self, words, tags=None):
        '''
        initializes this CRF instance.
        '''
        self.words = words
        self.tags = tags
        self.feature_cache = None

    def featurize(self, fv_uni):
        '''
        creates features (as list of strings) describing
        a tag unigram at a given position
        '''
        # fv_uni receives previous, current, and next word
        # and creates a list of features
        unigram_feats = []
        cur_word = None
        next_word = self.words[0]
        for i in xrange(len(self.words)):
            prev_word = cur_word
            cur_word = next_word
            try:
                next_word = self.words[i + 1]
            except IndexError:
                next_word = None
            feat = fv_uni(cur_word, prev_word, next_word)
            unigram_feats.append(feat)
        return unigram_feats

    def local_scores(self, unigram_feats, fc, learner, testing=True):
        '''
        computes scores and individual feature vectors
        for the unigram factors in the CRF
        '''
        uni_scores = numpy.zeros([len(self.words), len(tag_alphabet)])
        uni_fv = []
        for posn, feat in enumerate(unigram_feats):
            lbl_fv = []
            for i, lbl in enumerate(tag_alphabet):
                fv_hashed = fc.cross(feat, [lbl])
                lbl_fv.append(fv_hashed)
                score = learner.score(fv_hashed, testing)
                uni_scores[posn, i] = score
            uni_fv.append(lbl_fv)
        return (uni_scores, uni_fv)

    def augment_scores_with_loss(self, uni_scores):
        for (posn, tag) in enumerate(self.tags):
            lbl = tag_alphabet[tag]
            uni_scores[posn, :] += 1
            uni_scores[posn, lbl] -= 1

    def decode_scores(self, uni_scores, w_init, w_end, W_trans):
        '''
        does Viterbi decoding based on the unigram scores and
        init/end/transition weights
        '''
        forward_scores = numpy.zeros([len(self.words), len(tag_alphabet)])
        old_forward = w_init
        for (i, lbl_scores) in enumerate(uni_scores):
            # (ii) unigram scores
            old_forward += lbl_scores
            # print old_forward
            forward_scores[i] = old_forward
            # (i) best transition from a previous state
            old_forward = (W_trans + old_forward).max(1)
        old_forward = forward_scores[len(self.words) - 1] + w_end
        top_label = numpy.argmax(old_forward)
        result = self.decode_sequence(
            len(self.words) - 1, top_label,
            forward_scores, W_trans)
        result.reverse()
        assert len(result) == len(self.words), (result, self.words)
        return (result, old_forward[top_label])

    def decode_sequence(self, posn, label,
                        forward_scores, W_trans,
                        result=None):
        '''
        given a list of forward (Viterbi) scores, does a backtrace to
        recover the (reversed) optimal tag sequence
        '''
        if result is None:
            result = []
        result.append(tag_alphabet.get_sym(label))
        if posn > 0:
            backward = W_trans[label, :] + forward_scores[posn - 1]
            prev_label = numpy.argmax(backward)
            self.decode_sequence(posn - 1, prev_label,
                                 forward_scores, W_trans,
                                 result)
        return result

    def reconstruct_fv(self, uni_feats, fc, tags):
        '''
        given a complete output (tag sequence), gives
        the feature vectors associated with that pair
        of input and output
        '''
        Ntags = len(tag_alphabet)
        vectors = SparseVectorsD()
        tag_nums = [tag_alphabet[tag] for tag in tags]
        v_trans = [(tag_nums[0], 1.0),
                   (Ntags + tag_nums[-1], 1.0)]
        for i in xrange(len(tags)):
            uni_fv = uni_feats[i]
            fv_hashed = fc.cross(uni_fv, [tags[i]])
            vectors.add(fv_hashed)
            if i > 0:
                v_trans.append(
                    (Ntags * (2 + tag_nums[i]) + tag_nums[i - 1], 1.0))
        vectors.add(SparseVectorD(v_trans))
        return vectors

    def reconstruct_loss(self, tags):
        '''
        given a tag sequence, computes the loss wrt. the gold POS sequence.
        This has to be compatible with augment_score_with_loss
        '''
        result = 0.0
        for tag1, tag2 in izip(tags, self.tags):
            if tag1 != tag2:
                result += 1.0
        return result

    def local_rescore(self, uni_fv, learner, testing=True):
        '''
        Uses the cached feature vectors to compute
        the score faster than a full call of local_scores
        would be.
        '''
        Ntags = len(tag_alphabet)
        Nwords = len(self.words)
        uni_scores = numpy.zeros([Nwords, Ntags])
        for i, lbl_fv in enumerate(uni_fv):
            for j, fv_hashed in enumerate(lbl_fv):
                uni_scores[i, j] = learner.score(fv_hashed, testing)
        return uni_scores

    def make_training_example(self, fc, learner, verbose=False):
        '''
        computes the most-violating structure and returns the
        difference between gold and most-violating structure
        '''
        Ntags = len(tag_alphabet)
        if self.feature_cache is not None:
            (uni_feats, uni_fv) = self.feature_cache
            scores = self.local_rescore(uni_fv, learner, False)
        else:
            uni_feats = self.featurize(default_features)
            (scores, uni_fv) = self.local_scores(
                uni_feats, fc, learner, False)
            self.feature_cache = (uni_feats, uni_fv)
        self.augment_scores_with_loss(scores)
        w_init = learner.get_dense(0, Ntags, False)
        w_end = learner.get_dense(Ntags, 2 * Ntags, False)
        W_trans = learner.get_dense(
            2 * Ntags, (Ntags + 2) * Ntags, False).reshape(
                (Ntags, Ntags))
        (tags_bad, score_bad) = self.decode_scores(
            scores, w_init, w_end, W_trans)
        fv_good = self.reconstruct_fv(uni_feats, fc, self.tags)
        fv_bad = self.reconstruct_fv(uni_feats, fc, tags_bad)
        loss = self.reconstruct_loss(tags_bad)
        if verbose:
            # verify that featurize+decode_scores does that same thing
            # that reconstruct_fv and reconstruct_loss do
            # dump some explanatory info
            with LongLogEntry('progress', 'sample', loss=loss) as f:
                print >>f, "words:", self.words
                print >>f, "decoded:", tags_bad
                print >>f, "wanted:", self.tags
                for i in xrange(len(self.words)):
                    if tags_bad[i] != self.tags[i]:
                        print >>f, "  %s: %s (gld;%.3f) -> %s (sys;%.3f)" % (
                            self.words[i], self.tags[i],
                            scores[i][tag_alphabet[self.tags[i]]],
                            tags_bad[i],
                            scores[i][tag_alphabet[tags_bad[i]]])
            assert abs(learner.score(fv_bad.to_vec(), False) + loss - score_bad) < 0.01, (
                self.words, tags_bad,
                learner.score(fv_bad.to_vec(), False),
                loss, score_bad)
        if loss == 0.0:
            return (None, 0.0)
        else:
            result = fv_good - fv_bad
            return (result, loss)

    def decode(self, fc, learner):
        '''
        computes the best-scoring solution
        '''
        Ntags = len(tag_alphabet)
        features = self.featurize(default_features)
        (scores, uni_fv) = self.local_scores(features, fc, learner, True)
        w_init = learner.get_dense(0, Ntags, True)
        w_end = learner.get_dense(Ntags, 2 * Ntags, True)
        W_trans = learner.get_dense(
            2 * Ntags, (Ntags + 2) * Ntags, True).reshape(
                (Ntags, Ntags))
        (tags, score) = self.decode_scores(scores, w_init, w_end, W_trans)
        return tags


def train_epoch(crfs, fc, learner, epoch):
    t_start = time()
    shuffle(crfs)
    total_loss = 0.0
    for i, crf in enumerate(crfs):
        vec, loss = crf.make_training_example(
            fc, learner,
            verbose=(i % 1000 == 0))
        total_loss += loss
        if vec is not None:
            learner.update(vec, loss)
    t_end = time()
    add_log_item('progress', 'train_epoch',
                 epoch=epoch, seconds=t_end - t_start,
                 loss=total_loss, l1norm=learner.get_weights_l1())


def run_eval(crfs, fc, learner, epoch=None, out_fname=None):
    '''
    evaluates the learned weights on the given examples
    :param crfs the examples to evaluate
    :param fc   the feature hasher
    '''
    total_err = 0
    total_words = 0
    if out_fname is None:
        f_out = None
    else:
        f_out = file(out_fname, 'w')
    for crf in crfs:
        total_words += len(crf.words)
        tags = crf.decode(fc, learner)
        if f_out is not None:
            for i, w in enumerate(crf.words):
                print >>f_out, "%s\t%s"%(w, tags[i])
            print >>f_out
        total_err += crf.reconstruct_loss(tags)
    if f_out is not None:
        f_out.close()
    print "Error rate: %d/%d = %.3f" % (
        total_err, total_words, float(total_err) / total_words)
    add_log_item('progress', 'test_epoch',
                 epoch=epoch, total_err=total_err)
    return total_err

PATTERNS = []
for s_rg in [
        '[A-ZÄÖÜ]',
        '[a-zäöü]',
        '[0-9]+$']:
    PATTERNS.append(re.compile(s_rg))


def default_features(word, w_prev, w_next):
    '''
    basic features for POS tagging
    '''
    result = [
        'wc%s' % (word,),
        'wp%s' % (w_prev,),
        'wn%s' % (w_next,),
        'wP%s_%s' % (w_prev, word),
        'wN%s_%s' % (word, w_next)]
    for i in xrange(1, 5):
        result.append(
            'S%d%s' % (i, word[-i:]))
    for i, rgx in enumerate(PATTERNS):
        m = rgx.match(word)
        if m:
            result.append('r%d+' % (i,))
        else:
            result.append('r%d-' % (i,))
    return result

oparse = optparse.OptionParser()
oparse.add_option('--dev', dest='dev_fname')
oparse.add_option('--l2', dest='l2',
                  default=0.0, type='float')
oparse.add_option('--learner', dest='learner',
                   default='sgd_momentum',
                   choices=['sgd_momentum', 'mira'])
oparse.add_option('--epochs', dest='num_epochs',
                  default=50, type='int')
oparse.add_option('--prefix', dest='prefix',
                  default='linear_crf')

oparse_test = optparse.OptionParser()
oparse_test.add_option('--prefix', dest='prefix',
                       default='linear_crf')

# assume CoNLL-X format
WORD_COLUMN = 1
TAG_COLUMN = 4

def train_crf_main(argv=None):
    '''
    run complete training loop, and output predictions
    after each 10 iterations or so
    '''
    opts, args = oparse.parse_args(argv)
    set_logfile_prefix(opts.prefix)
    fname = args[0]
    trees = list(read_tabular(fname, [WORD_COLUMN, TAG_COLUMN]))
    for words, tags in trees:
        for tag in tags:
            tag_alphabet[tag]
    tag_alphabet.tofile(file(opts.prefix+'_tags.txt', 'w'))
    crfs = [LinearCRF(words, tags)
            for (words, tags) in trees]
    print "Read %d training examples." % (len(crfs),)
    fc = FeatureHasher()
    if opts.dev_fname is not None:
        dev_trees = list(read_tabular(opts.dev_fname, [WORD_COLUMN, TAG_COLUMN]))
        dev_crfs = [LinearCRF(words, tags)
                    for (words, tags) in dev_trees]
        print "Read %d dev examples" % (len(dev_crfs),)
    else:
        dev_crfs = []
    learner = create_learner(opts.learner, fc.num_dimensions,
                             n_examples=len(crfs), l2=opts.l2)
    best_loss = 1e9
    for i in xrange(opts.num_epochs):
        print >>sys.stderr, "EPOCH:", i
        train_epoch(crfs, fc, learner, epoch=i)
        if i % 2 == 0 and i >= 3 and dev_crfs:
            loss = run_eval(dev_crfs, fc, learner, epoch=i,
                            out_fname='%s_dev_epoch%d.txt'%(opts.prefix, i))
            if loss < best_loss:
                learner.save_binary(opts.prefix+'_weights.npy')
                copyfile('%s_dev_epoch%d.txt'%(opts.prefix, i),
                         '%s_dev_best.txt'%(opts.prefix,))
    learner.save_binary('tagger_weights.npy')

def test_crf_main(argv=None):
    '''
    runs the CRF tagger on a file and reports
    the accuracy.
    '''
    opts, args = oparse_test.parse_args(argv)
    fc = FeatureHasher()
    learner = load_weights(opts.prefix+'_weights.npy')
    tag_alphabet.fromfile(file(opts.prefix+'_tags.txt'))
    total_words = 0
    total_err = 0
    f_out = sys.stdout
    for words, tags in read_tabular(args[0], [WORD_COLUMN, TAG_COLUMN]):
        crf = LinearCRF(words, tags)
        total_words += len(crf.words)
        tags = crf.decode(fc, learner)
        if f_out is not None:
            for i, w in enumerate(crf.words):
                print >>f_out, "%s\t%s"%(w, tags[i])
            print >>f_out
        total_err += crf.reconstruct_loss(tags)
    print >>sys.stderr, "Total error: %d/%d = %.3f"%(total_err, total_words,
                                                     float(total_err)/total_words)




if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test']:
        print >>sys.stderr, "Usage: %s (train|test) file..." % (sys.argv[0],)
        sys.exit(1)
    if sys.argv[1] == 'train':
        train_crf_main(sys.argv[2:])
    elif sys.argv[1] == 'test':
        test_crf_main(sys.argv[2:])
