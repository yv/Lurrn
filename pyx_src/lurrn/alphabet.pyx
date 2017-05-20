cimport numpy


cdef class AbstractAlphabet:
    cdef int size(self):
        return 0
    cdef int sym2num(self,const_char_ptr sym):
        return -1
    cdef const_char_ptr num2sym(self, int num):
        return NULL
    cpdef get_sym(self, int num):
        cdef const_char_ptr res=self.num2sym(num)
        if res is NULL:
            raise IndexError
        else:
            return res
    def tofile(self,f,fmt=1):
        cdef int i, n
        n=self.size()
        w=f.write
        if fmt in [0,1]:
            if fmt==1:
                w(b'%d\n'%(n,))
            for i from 0<=i<n:
                w(bytes(self.num2sym(i)))
                w(b'\n')
    def fromfile(self,f,fmt=1):
        cdef int i, n
        rl=f.readline
        if fmt==0:
            while True:
                s=rl()
                if s=='':
                    break
                s=s.rstrip(b'\n')
                self.sym2num(s)
        elif fmt==1:
            n=int(rl().rstrip(b'\n'))
            for i from 0<=i<n:
                s=rl()
                s=s.rstrip(b'\n')
                self.sym2num(s)

cdef class Alphabet_iter:
    def __init__(self,alph):
        self.alph=alph
    def __next__(self):
        if self.pos>=self.alph.size():
            raise StopIteration
        else:
            res=self.alph.num2sym(self.pos)
            self.pos+=1
            return res

cdef class StringAlphabet

cdef class Alphabet_UTF8_iter(Alphabet_iter):
    def __init__(self,alph):
        self.alph=alph
    def __next__(self):
        cdef const_char_ptr s
        if self.pos>=self.alph.size():
            raise StopIteration
        else:
            s=self.alph.num2sym(self.pos)
            res=PyUnicode_DecodeUTF8(s,strlen(s),"ignore")
            self.pos+=1
            return res

cdef class Alphabet_Latin1_iter(Alphabet_iter):
    def __init__(self,alph):
        self.alph=alph
    def __next__(self):
        cdef const_char_ptr s
        if self.pos>=self.alph.size():
            raise StopIteration
        else:
            s=self.alph.num2sym(self.pos)
            res=PyUnicode_DecodeLatin1(s,strlen(s),"ignore")
            self.pos+=1
            return res

cdef class PythonAlphabet(AbstractAlphabet):
    def __init__(self):
        self.mapping={}
        self.words=[]
        self.growing=True
    cdef int size(self):
        return PyList_Size(self.words)
    def __len__(self):
        return self.size()
    cdef int sym2num(self, const_char_ptr sym):
        cdef int retval
        cdef PyObject *val=PyDict_GetItemString(self.mapping,sym)
        if val is NULL:
            if self.growing:
                retval=PyList_Size(self.words)
                s=sym
                PyList_Append(self.words,s)
                PyDict_SetItem(self.mapping,s,retval)
                return retval
            else:
                return -1
        else:
            return <object>val
    cdef const_char_ptr num2sym(self, int num):
        cdef object val
        cdef const_char_ptr res
        val=<object>PyList_GetItem(self.words,num)
        res=val
        return res
    cpdef get_sym(self,int num):
        return self.words[num]
    cpdef remap(self, numpy.ndarray filt_array):
        cdef int i,n
        n=len(self.words)
        alph2=PythonAlphabet()
        for i from 0<=i<n:
            if filt_array[i]:
                alph2[self.words[i]]
        alph2.growing=self.growing
        return alph2
    def __contains__(self, word):
        cdef PyObject *val=PyDict_GetItem(self.mapping,word)
        if val is NULL:
            return False
        else:
            return True
    def __iter__(self):
        return iter(self.words)
    def __getitem__(self, word):
        cdef int retval
        cdef PyObject *val=PyDict_GetItem(self.mapping,word)
        if val is NULL:
            if self.growing:
                retval=PyList_Size(self.words)
                PyList_Append(self.words,word)
                PyDict_SetItem(self.mapping,word,retval)
                return retval
            else:
                raise KeyError(word)
        else:
            return <object>val
    def __reduce__(self):
        # choose compatibility over performance.
        return (PythonAlphabet,(),self.words)
    def __setstate__(self,words):
        self.words=words
        self.mapping={}
        for i,w in enumerate(self.words):
            self.mapping[w]=i
        self.growing=False

cdef class StringAlphabet(AbstractAlphabet):
    """
    provides an Alphabet that can store unicode and bytes objects,
    where unicode objects are stored as (by default) UTF-8 encoded strings
    """
    def __cinit__(self, want_utf8=True):
        self.use_utf8=want_utf8
    property growing:
        def __get__(self):
            return self.map.growing
        def __set__(self,val):
            self.map.growing=val
    cdef int sym2num(self,const_char_ptr sym):
        return self.map.sym2num(sym)
    cdef const_char_ptr num2sym(self, int num):
        return self.map.num2sym(num)
    cpdef remap(self, numpy.ndarray filt_array):
        cdef int i,n
        cdef StringAlphabet alph2=StringAlphabet()
        n=self.map.size()
        for i from 0<=i<n:
            if filt_array[i]:
                alph2.map.sym2num(self.map.num2sym(i))
        alph2.map.growing=self.map.growing
        return alph2
    cdef int size(self):
        return self.map.size()
    def __len__(self):
        return self.map.size()

    def get_sym_unicode(self, int num):
        cdef const_char_ptr res=self.num2sym(num)
        if res is NULL:
            raise IndexError
        else:
            if self.use_utf8:
                resObj=PyUnicode_DecodeUTF8(res,strlen(res),"ignore")
            else:
                resObj=PyUnicode_DecodeLatin1(res,strlen(res),"ignore")
            return resObj
    def __iter__(self):
        if self.use_utf8:
            return Alphabet_UTF8_iter(self)
        else:
            return Alphabet_Latin1_iter(self)
    def __getitem__(self,sym):
        cdef bytes sym_s
        if PyUnicode_Check(sym):
            if self.use_utf8:
                sym_s=PyUnicode_AsUTF8String(sym)
            else:
                try:
                    sym_s=PyUnicode_AsLatin1String(sym)
                except UnicodeEncodeError:
                    sym_o=PyUnicode_AsEncodedString(sym,"ISO-8859-15","replace")
                    sym_s=sym_o
        else:
            sym_s=sym
        cdef int res=self.sym2num(sym_s)
        if res==-1:
            raise KeyError
        else:
            return res
    def __contains__(self, sym):
        cdef bytes sym_s
        if PyUnicode_Check(sym):
            if self.use_utf8:
                sym_s=PyUnicode_AsUTF8String(sym)
            else:
                try:
                    sym_s=PyUnicode_AsLatin1String(sym)
                except UnicodeEncodeError:
                    sym_o=PyUnicode_AsEncodedString(sym,"ISO-8859-15","replace")
                    sym_s=sym_o
        else:
            sym_s=sym
        cdef int res=self.sym2num(sym_s)
        if res==-1:
            return False
        else:
            return True

    def tofile_utf8(self,f,fmt=1):
        cdef int i, n
        n=self.size()
        if self.use_utf8:
            self.tofile(f,fmt)
            return
        w=f.write
        if fmt in [0,1]:
            if fmt==1:
                w(b'%d\n'%(n,))
            for i from 0<=i<n:
                w(self.get_sym_unicode(i).encode('UTF-8'))
                w(b'\n')
    def fromfile_utf8(self,f,fmt=1):
        cdef int i, n
        rl=f.readline
        if self.use_utf8:
            self.fromfile(f,fmt)
            return
        if fmt==0:
            while True:
                s=rl()
                if s=='':
                    break
                s=s.rstrip(b'\n')
                self[s.decode('UTF-8')]
        elif fmt==1:
            n=int(rl().rstrip('\n'))
            for i from 0<=i<n:
                s=rl()
                s=s.rstrip(b'\n')
                self[s.decode('UTF-8')]

