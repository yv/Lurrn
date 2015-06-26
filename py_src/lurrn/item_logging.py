import sys
import os
from cStringIO import StringIO
from collections import defaultdict
from time import clock, ctime
import traceback
import pdb

rate_limits = {}
next_output = defaultdict(float)
skipped = defaultdict(int)
messages = {}

logfile = sys.stderr

def debug_on_error(tp, value, tb):
    if (not sys.stderr.isatty() or
        not sys.stdin.isatty()):
        log_exception(traceback.format_exception(tp, value, tb))
        original_hook(tp, value, tb)
    else:
        traceback.print_exception(tp, value, tb)
        print
        pdb.pm()

original_hook = sys.excepthook

def install_exception_handler():
    sys.excepthook = debug_on_error

def set_rate_limit(topic, min_sec):
    rate_limits[topic] = min_sec


def set_message(topic, event, s):
    messages[(topic, event)] = s


def add_log_item(topic, event, **kwargs):
    if topic in rate_limits:
        tm = clock()
        if tm < next_output[topic]:
            skipped[topic] += 1
            return
    else:
        tm = None
    if (topic, event) in messages:
        s = messages[(topic, event)] % kwargs
    else:
        s = '[%s %s]' % (
            event,
            ' '.join(['%s:%s' % (k, v) for (k, v) in kwargs.iteritems()]))
    n_skipped = skipped[topic]
    if n_skipped:
        s_skipped = ' [skipped %d]' % (n_skipped,)
        skipped[topic] = 0
    else:
        s_skipped = ''
    print >>logfile, "%s %s %s%s" % (ctime(), topic, s, s_skipped)
    logfile.flush()
    if tm is not None:
        next_output[topic] = tm + rate_limits[topic]

def log_exception(exception_text, topic='ERROR'):
    if type(exception_text) == list:
        exception_text = ''.join(exception_text)
    print >>logfile, "%s %s [[[exception"%(ctime(), topic)
    print >>logfile, exception_text
    print >>logfile, "]]]"

def set_logfile(fname):
    global logfile
    logfile.flush()
    logfile = file(fname, 'a')

def set_logfile_prefix(prefix):
    fname = '%s_%s.log'%(prefix, os.getpid())
    set_logfile(fname)

class LongLogEntry:
    def __init__(self, topic='INFO', event='description', **kwargs):
        if (topic, event) in messages:
            s = messages[(topic, event)] % kwargs
        else:
            s = '[%s %s]' % (
                event,
                ' '.join(['%s:%s' % (k, v) for (k, v) in kwargs.iteritems()]))
        self.topic = topic
        self.s = s
        self.kwargs = kwargs
        self.stream = StringIO()
    def __enter__(self):
        return self.stream
    def __exit__(self, type, value, traceback):
        topic = self.topic
        if topic in rate_limits:
            tm = clock()
            if tm < next_output[topic]:
                skipped[topic] += 1
                return
        else:
            tm = None
        n_skipped = skipped[topic]
        if n_skipped:
            s_skipped = ' [skipped %d]' % (n_skipped,)
            skipped[topic] = 0
        else:
            s_skipped = ''
        print >>logfile, "%s %s %s%s [[["%(ctime(), topic, self.s, s_skipped)
        print >>logfile, self.stream.getvalue(),
        print >>logfile, "]]]"
