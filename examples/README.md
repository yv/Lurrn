Examples
========

Linear CRF
----------

this provides a linear CRF as you could also get it from, e.g. CRF++.
If your training and development files are train.conll and dev.conll,
respectively, you can do a

python linear_crf.py train train.conll --dev dev.conll --l2 1.0

and subsequently

python linear_crf.py test dev.conll

