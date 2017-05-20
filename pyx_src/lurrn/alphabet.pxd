cimport numpy

cdef extern from "string.h":
    size_t strlen(char *s)

cdef extern from *:
    ctypedef char* const_char_ptr "const char*"
    void cxx_delete "delete" (void *)

from cpython cimport *

cdef extern from "cxx_alph.h":
    cppclass c_CPPAlphabet "CPPAlphabet":
        int size()
        int sym2num(const_char_ptr sym)
        const_char_ptr num2sym(int num)
        bint growing
        void call_destructor "~CPPAlphabet" ()

cdef class AbstractAlphabet:
    cdef int size(self)
    cdef int sym2num(self,const_char_ptr sym)
    cdef const_char_ptr num2sym(self, int num)
    cpdef get_sym(self, int num)

cdef class Alphabet_iter:
   cdef AbstractAlphabet alph
   cdef int pos

cdef class PythonAlphabet(AbstractAlphabet):
    cdef public object mapping
    cdef public object words
    cdef public bint growing
    cdef int size(self)
    cdef int sym2num(self, const_char_ptr sym)
    cdef const_char_ptr num2sym(self, int num)
    cpdef remap(self, numpy.ndarray filt_array)

cdef class StringAlphabet(AbstractAlphabet):
    cdef bint use_utf8
    cdef c_CPPAlphabet map
    cdef int size(self)
    cdef int sym2num(self,const_char_ptr sym)
    cdef const_char_ptr num2sym(self, int num)
    cpdef remap(self, numpy.ndarray filt_array)

